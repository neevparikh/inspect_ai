import dataclasses
import functools
import json
import time
import uuid
from abc import abstractmethod
from logging import getLogger
from typing import Any, Generic, TypeVar

import anyio
import anyio.abc

from inspect_ai._util._async import run_in_background, tg_collect
from inspect_ai._util.constants import DEFAULT_BATCH_SIZE
from inspect_ai._util.notgiven import sanitize_notgiven
from inspect_ai.model._generate_config import BatchConfig, GenerateConfig

DEFAULT_BATCH_TICK = 15
DEFAULT_SEND_DELAY = DEFAULT_BATCH_TICK
DEFAULT_MAX_BATCHES = 50
MAX_CONSECUTIVE_CHECK_FAILURES = 1000

logger = getLogger(__name__)

ResponseT = TypeVar("ResponseT")
CompletedBatchInfoT = TypeVar("CompletedBatchInfoT")
"""
This is model provider specific info that represents the completed result of a batch

It gets returned by the `_check_batch` method and passed to `_handle_batch_result`.

Not all model providers need this
"""


@dataclasses.dataclass
class BatchRequest(Generic[ResponseT]):
    """This is a single request that is part of a batch."""

    request: dict[str, Any]
    result_stream: anyio.abc.ObjectSendStream[ResponseT | Exception]
    custom_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))


@dataclasses.dataclass
class Batch(Generic[ResponseT]):
    id: str
    requests: dict[str, BatchRequest[ResponseT]]
    consecutive_check_failure_count: int = 0
    completed_count: int = 0
    failed_count: int = 0


@dataclasses.dataclass
class PendingBatch(Generic[ResponseT]):
    timeout: float
    available_size: int
    requests: list[BatchRequest[ResponseT]] = dataclasses.field(default_factory=list)


class Batcher(Generic[ResponseT, CompletedBatchInfoT]):
    def __init__(
        self,
        config: BatchConfig,
        max_batch_request_count: int,
        max_batch_size_mb: int,
    ) -> None:
        # self.config = config
        self._max_batch_request_count = max_batch_request_count
        self._max_batch_size_bytes = max_batch_size_mb * 1024 * 1024
        self._min_batch_request_count = config.size or DEFAULT_BATCH_SIZE
        self._send_delay = config.send_delay or DEFAULT_SEND_DELAY
        self._tick = config.tick or DEFAULT_BATCH_TICK
        self._max_batches = config.max_batches or DEFAULT_MAX_BATCHES
        self._intake_queue: list[BatchRequest[ResponseT]] = []
        self._next_batch: PendingBatch[ResponseT] | None = None
        self._inflight_batches: dict[str, Batch[ResponseT]] = {}
        self._is_batch_worker_running: bool = False

    async def generate(
        self, request: dict[str, Any], config: GenerateConfig
    ) -> ResponseT:
        send_stream, receive_stream = anyio.create_memory_object_stream[
            ResponseT | Exception
        ](1)
        batch_request = BatchRequest[ResponseT](
            request=request, result_stream=send_stream
        )
        self._intake_queue.append(batch_request)

        if not self._is_batch_worker_running:
            self._is_batch_worker_running = True
            run_in_background(self._batch_worker)

        result = await receive_stream.receive()
        if isinstance(result, Exception):
            raise result
        return result

    async def _batch_worker(self) -> None:
        from inspect_ai.log._transcript import Transcript, init_transcript

        init_transcript(Transcript())

        try:
            while self._is_there_work_to_do():
                await self._check_inflight_batches()

                while await self._process_intake_queue():
                    pass

                await anyio.sleep(self._tick)

            logger.info("Batch worker finished processing...exiting")
            self._is_batch_worker_running = False
        finally:
            logger.info("Batch worker finally...exiting")

    def _is_there_work_to_do(self) -> bool:
        result = bool(
            self._inflight_batches
            or self._intake_queue
            or (self._next_batch.requests if self._next_batch else False)
        )
        logger.info(
            f"_is_there_work_to_do: {result} inflight batches: {len(self._inflight_batches)} intake queue: {len(self._intake_queue)} _next_batch.requests: {len(self._next_batch.requests) if self._next_batch else 0}"
        )
        return result

    async def _check_inflight_batches(self) -> None:
        if self._inflight_batches:
            await tg_collect(
                [
                    functools.partial(self._check_inflight_batch, batch)
                    for batch in self._inflight_batches.values()
                ]
            )

        total_requests, total_completed, total_failed = functools.reduce(
            lambda acc, batch: (
                acc[0] + len(batch.requests),
                acc[1] + batch.completed_count,
                acc[2] + batch.failed_count,
            ),
            self._inflight_batches.values(),
            (0, 0, 0),
        )
        logger.info(
            f"Inflight batches: {len(self._inflight_batches)}, "
            f"pending/completed/failed requests: {total_requests - total_completed - total_failed}/{total_completed}/{total_failed}"
        )

    async def _check_inflight_batch(self, batch: Batch[ResponseT]) -> None:
        check_result = await self._wrapped_check_batch(batch)
        if not check_result:
            return

        completed_requests, failed_requests, completed_info = check_result
        batch.completed_count = completed_requests
        batch.failed_count = failed_requests
        if completed_info is None:
            return

        await self._wrapped_handle_batch_result(batch, completed_info)

    async def _fail_and_cleanup_inflight_batch(
        self,
        batch: Batch[ResponseT],
        error: Exception,
    ) -> None:
        await self._fail_all_requests(list(batch.requests.values()), error)
        del self._inflight_batches[batch.id]

    async def _fail_all_requests(
        self, batch_requests: list[BatchRequest[ResponseT]], error: Exception
    ) -> None:
        for request in batch_requests:
            try:
                await request.result_stream.send(
                    error or self._get_request_failed_error(request)
                )
            except anyio.BrokenResourceError:
                # Stream closed (client disconnected/completed) - continue
                # notifying remaining requests
                pass

    async def _process_intake_queue(self) -> bool:
        """Process intake queue and send next batch if conditions are met."""
        if self._next_batch is None:
            self._next_batch = PendingBatch(
                time.time() + self._send_delay,
                int(self._max_batch_size_bytes * 0.95),
            )

        add_count, new_avail, should_send = _assess_intake_queue(
            self._intake_queue,
            self._next_batch,
            self._min_batch_request_count,
            self._max_batch_request_count,
        )

        if add_count:
            self._next_batch = PendingBatch(
                self._next_batch.timeout,
                new_avail,
                self._next_batch.requests + self._intake_queue[:add_count],
            )
            self._intake_queue = self._intake_queue[add_count:]

        if should_send and len(self._inflight_batches) < self._max_batches:
            batch_requests = self._next_batch.requests
            self._next_batch = None

            batch_id = await self._wrapped_create_batch(batch_requests)

            self._inflight_batches[batch_id] = Batch(
                id=batch_id,
                requests={request.custom_id: request for request in batch_requests},
            )
            return True

        return False

    # These _wrapped_* methods are intended to wrap the abstract methods with the
    # appropriate error handling logic consistent with the batch algorithm. This
    # allows the code above to not worry about try/catch'ing the abstract methods.
    # Any exception that escapes a _wrapped_* method will bring down the eval.

    async def _wrapped_create_batch(self, batch: list[BatchRequest[ResponseT]]) -> str:
        try:
            result = await self._create_batch(batch)
            logger.info(f"Created batch {result} with {len(batch)} requests")
            return result
        except Exception as e:
            logger.info(
                f"Error creating batch, failing all {len(batch)} requests in batch. Error: {e}"
            )
            await self._fail_all_requests(batch, e)
            raise

    async def _wrapped_check_batch(
        self, batch: Batch[ResponseT]
    ) -> tuple[int, int, (CompletedBatchInfoT | None)] | None:
        try:
            result = await self._check_batch(batch)
            batch.consecutive_check_failure_count = 0
            return result
        except Exception as e:
            logger.error(f"Error checking batch {batch.id}", exc_info=e)
            batch.consecutive_check_failure_count += 1
            if batch.consecutive_check_failure_count >= MAX_CONSECUTIVE_CHECK_FAILURES:
                logger.error(
                    f"Batch {batch.id} failed after {MAX_CONSECUTIVE_CHECK_FAILURES} retries, failing all {len(batch.requests)} requests in batch",
                )
                await self._fail_and_cleanup_inflight_batch(batch, e)
            return None

    async def _wrapped_handle_batch_result(
        self,
        batch: Batch[ResponseT],
        completion_info: CompletedBatchInfoT,
    ) -> None:
        try:
            results = await self._handle_batch_result(batch, completion_info)
            if len(results) != len(batch.requests):
                logger.error(
                    f"Batch {batch.id} returned {len(results)} results, expected {len(batch.requests)}",
                )
            for request_id, response in results.items():
                await batch.requests[request_id].result_stream.send(response)
            del self._inflight_batches[batch.id]
        except Exception as e:
            logger.info(
                f"Batch {batch.id} failed after retries, failing all {len(batch.requests)} requests in batch",
            )
            await self._fail_and_cleanup_inflight_batch(batch, e)

    @abstractmethod
    async def _create_batch(self, batch: list[BatchRequest[ResponseT]]) -> str:
        pass

    @abstractmethod
    async def _check_batch(
        self, batch: Batch[ResponseT]
    ) -> tuple[int, int, (CompletedBatchInfoT | None)]:
        """Returns the number of completed requests, failed requests, and completed batch info if completed"""
        pass

    @abstractmethod
    async def _handle_batch_result(
        self,
        batch: Batch[ResponseT],
        completion_info: CompletedBatchInfoT,
    ) -> dict[str, ResponseT | Exception]:
        pass

    @abstractmethod
    # Must not let any exceptions escape. Any exception that does escape is a
    # coding error and will bring down the eval.
    def _get_request_failed_error(self, request: BatchRequest[ResponseT]) -> Exception:
        pass


def _assess_intake_queue(
    intake_queue: list[BatchRequest[ResponseT]],
    batch: PendingBatch[ResponseT],
    min_request_count: int,
    max_request_count: int,
) -> tuple[int, int, bool]:
    """Assess the intake queue and determine what should be done with the current batch.

    This function determines two things:

    1. How many (if any) requests from the `intake_queue` can be added to `batch`.
       This is constrained by `batch.available_size` and `max_batch_request_count`
       - neither of which can be exceeded.

    2. Whether the resulting/post-add batch should be sent now or not. This will
       be `True` if the post-add batch is:
       - full - either request count or bytes
       - has at least `min_batch_request_count` requests
       - has waited until `batch.timeout` to send the batch

    At a high level, the algorithm endeavors to add as many requests as possible
    from the `intake_queue` to the `batch`, while respecting all constraints.

    Args:
        intake_queue: List of batch requests waiting to be processed
        batch: Current batch being assembled
        min_request_count: Minimum number of requests before sending
        max_request_count: Maximum number of requests allowed in a batch

    Returns:
        A tuple of (add_count, new_available_size, should_send) where:
        - add_count: Number of requests to add from intake_queue to pending_batch
        - new_available_size: Remaining available size in bytes after adding requests
        - should_send: Whether the batch should be sent now
    """
    add_count = 0
    current_count = len(batch.requests)
    available_count = max_request_count - current_count
    available_size = batch.available_size
    batch_full = available_count <= 0 or available_size <= 0

    for request in intake_queue:
        if batch_full:
            break

        # TODO: DO NOT MERGE
        # This is just for debugging to allow breaking into multiple batches
        # if current_count + add_count >= min_request_count:
        #     break

        request_size = len(
            json.dumps(sanitize_notgiven(request.request), separators=(",", ":"))
        )

        if request_size > available_size:
            if current_count + add_count == 0:
                raise ValueError(
                    f"Single request size {request_size} exceeds maximum size {available_size}."
                )
            batch_full = True
        else:
            # Request fits, add it
            add_count += 1
            available_size -= request_size
            available_count -= 1
            batch_full = available_count <= 0

    should_send = (
        batch_full
        or ((new_count := current_count + add_count) >= min_request_count)
        or (time.time() > batch.timeout and new_count > 0)
    )

    return add_count, available_size, should_send
