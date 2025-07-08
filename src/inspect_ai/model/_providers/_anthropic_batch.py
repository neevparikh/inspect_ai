import random
from typing import TypeAlias, cast

import httpx
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    AsyncAnthropicBedrock,
    AsyncAnthropicVertex,
    InternalServerError,
)
from anthropic.types import (
    APIErrorObject,
    AuthenticationError,
    BillingError,
    GatewayTimeoutError,
    InvalidRequestError,
    Message,
    NotFoundError,
    OverloadedError,
    RateLimitError,
)
from anthropic.types import (
    PermissionError as AnthropicPermissionError,
)
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import (
    Request as AnthropicBatchRequest,
)
from tenacity import retry

from inspect_ai.model._generate_config import BatchConfig
from inspect_ai.model._retry import ModelRetryConfig

from .util.batch import (
    Batch,
    Batcher,
    BatchRequest,
)
from .util.hooks import HttpxHooks

CompletedBatchInfo: TypeAlias = bool


class AnthropicBatcher(Batcher[Message, CompletedBatchInfo]):
    def __init__(
        self,
        client: AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicVertex,
        config: BatchConfig,
        retry_config: ModelRetryConfig,
    ):
        super().__init__(
            config,
            max_batch_request_count=100000,
            max_batch_size_mb=256,
        )
        self._client = client
        self._retry_config = retry_config

    async def _create_batch(self, batch: list[BatchRequest[Message]]) -> str:
        @retry(**self._retry_config)
        async def _create() -> str:
            requests: list[AnthropicBatchRequest] = []
            extra_headers: dict[str, str] = {}
            for request in batch:
                extra_headers = request.request.pop("extra_headers", {})
                request_id = extra_headers.pop(HttpxHooks.REQUEST_ID_HEADER, None)
                if request_id is not None:
                    request.custom_id = request_id
                requests.append(
                    AnthropicBatchRequest(
                        custom_id=request.custom_id,
                        params=cast(MessageCreateParamsNonStreaming, request.request),
                    )
                )

            # TODO: DON'T MERGE
            # Randomly raise an error for 1 in 5 requests
            if random.randint(1, 3) == 1:
                raise APITimeoutError(
                    request=httpx.Request(
                        method="POST",
                        url="https://api.anthropic.com/v1/messages/batches",
                    )
                )

            batch_info = await self._client.messages.batches.create(
                requests=requests,
                extra_headers=extra_headers or None,
            )
            return batch_info.id

        return await _create()

    async def _check_batch(self, batch: Batch[Message]) -> CompletedBatchInfo | None:
        batch_info = await self._client.messages.batches.retrieve(batch.id)

        # Only show non-zero counts
        counts = []
        if batch_info.request_counts.processing > 0:
            counts.append(f"processing={batch_info.request_counts.processing}")
        if batch_info.request_counts.expired > 0:
            counts.append(f"expired={batch_info.request_counts.expired}")
        if batch_info.request_counts.canceled > 0:
            counts.append(f"canceled={batch_info.request_counts.canceled}")
        if batch_info.request_counts.errored > 0:
            counts.append(f"errored={batch_info.request_counts.errored}")
        if batch_info.request_counts.succeeded > 0:
            counts.append(f"succeeded={batch_info.request_counts.succeeded}")

        counts_str = ", ".join(counts) if counts else "no active requests"
        print(
            f"Checking batch {batch.id}: {batch_info.processing_status} ({counts_str})"
        )

        # We don't need any extra completion info beyond the True since we
        # retrieve the results directly via the sdk given the batch id.
        return True if batch_info.processing_status == "ended" else None

    async def _handle_batch_result(
        self,
        batch: Batch[Message],
        completion_info: CompletedBatchInfo,
    ) -> None:
        import anthropic

        async for result in await self._client.messages.batches.results(batch.id):
            custom_id = result.custom_id
            batch_request = batch.requests.pop(custom_id)

            response: Message | Exception
            match result.result.type:
                case "succeeded":
                    response = result.result.message
                case "errored":
                    # See anthropic._client.AsyncAnthropic._make_status_error
                    message = result.result.error.error.message
                    error_class: type[anthropic.APIStatusError]
                    match result.result.error.error:
                        case InvalidRequestError():
                            error_class = anthropic.BadRequestError
                        case AuthenticationError():
                            error_class = anthropic.AuthenticationError
                        case BillingError():
                            error_class = anthropic.PermissionDeniedError
                        case AnthropicPermissionError():
                            error_class = anthropic.PermissionDeniedError
                        case NotFoundError():
                            error_class = anthropic.NotFoundError
                        case RateLimitError():
                            error_class = anthropic.RateLimitError
                        case GatewayTimeoutError():
                            error_class = anthropic.InternalServerError
                        case APIErrorObject():
                            error_class = anthropic.APIStatusError
                        case OverloadedError():
                            error_class = anthropic.InternalServerError
                    response = error_class(
                        message=message,
                        response=httpx.Response(status_code=500, text=message),
                        body=None,
                    )
                    response.response.status_code = response.status_code
                case "canceled":
                    response = APIConnectionError(
                        request=httpx.Request(
                            method="POST",
                            url="https://api.anthropic.com/v1/messages/batches",
                        )
                    )
                case "expired":
                    response = APITimeoutError(
                        request=httpx.Request(
                            method="POST",
                            url="https://api.anthropic.com/v1/messages/batches",
                        )
                    )

            await batch_request.result_stream.send(response)

    def _get_request_failed_error(self, request: BatchRequest[Message]) -> Exception:
        return InternalServerError(
            message="Request failed",
            response=httpx.Response(
                status_code=500,
                text="Request failed",
                request=httpx.Request(
                    method="POST",
                    url="https://api.anthropic.com/v1/messages/batches",
                ),
            ),
            body=None,
        )
