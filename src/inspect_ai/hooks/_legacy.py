"""Legacy hooks for telemetry and API key overrides.

These are deprecated and will be removed in a future release. Please use the new hooks
defined in `inspect_ai.hooks` instead.
"""

import importlib
import os
from typing import Any, Awaitable, Callable, Literal, cast

from inspect_ai._util.error import PrerequisiteError

# Hooks are functions inside packages that are installed with an
# environment variable (e.g. INSPECT_TELEMETRY='mypackage.send_telemetry')
# If one or more hooks are enabled a message will be printed at startup
# indicating this, as well as which package/function implements each hook


# Telemetry (INSPECT_TELEMETRY)
#
# Telemetry can be optionally enabled by setting an INSPECT_TELEMETRY
# environment variable that points to a function in a package which
# conforms to the TelemetrySend signature below. A return value of True
# indicates that the telemetry event was handled.

# There are currently three types of telemetry sent:
#    - model_usage       (JSON string of the model usage)
#    - eval_log_location (file path or URL string of the eval log)
#    - eval_log          (JSON string of the eval log)
#                        [only sent if eval_log_location unhandled]
# The eval_log_location type is preferred over eval_log as it means we can take
# advantage of the .eval format and avoid loading the whole log into memory.

TelemetrySend = Callable[[str, str], Awaitable[bool]]


async def send_telemetry_legacy(
    type: Literal["model_usage", "eval_log", "eval_log_location"], json: str
) -> Literal["handled", "not_handled", "no_subscribers"]:
    global _send_telemetry
    if _send_telemetry:
        if await _send_telemetry(type, json):
            return "handled"
        return "not_handled"
    return "no_subscribers"


_send_telemetry: TelemetrySend | None = None

# API Key Override (INSPECT_API_KEY_OVERRIDE)
#
# API Key overrides can be optionally enabled by setting an
# INSPECT_API_KEY_OVERRIDE environment variable which conforms to the
# ApiKeyOverride signature below.
#
# The api key override function will be called with the name and value
# of provider specified environment variables that contain api keys,
# and it can optionally return an override value.

ApiKeyOverride = Callable[[str, str], str | None]


def override_api_key_legacy(var: str, value: str) -> str | None:
    global _override_api_key
    if _override_api_key:
        return _override_api_key(var, value)
    else:
        return None


_override_api_key: ApiKeyOverride | None = None


def init_legacy_hooks() -> list[str]:
    messages: list[str] = []
    # telemetry
    global _send_telemetry
    if not _send_telemetry:
        result = init_legacy_hook(
            "telemetry",
            "INSPECT_TELEMETRY",
            "(eval logs and token usage will be recorded by the provider)",
        )
        if result:
            _send_telemetry, message = result
            messages.append(message)

    # api key override
    global _override_api_key
    if not _override_api_key:
        result = init_legacy_hook(
            "api key override",
            "INSPECT_API_KEY_OVERRIDE",
            "(api keys will be read and modified by the provider)",
        )
        if result:
            _override_api_key, message = result
            messages.append(message)

    return messages


def init_legacy_hook(
    name: str, env: str, message: str
) -> tuple[Callable[..., Any], str] | None:
    hook = os.environ.get(env, "")
    if hook:
        # parse module/function
        module_name, function_name = hook.strip().rsplit(".", 1)
        # load (fail gracefully w/ clear error)
        try:
            module = importlib.import_module(module_name)
            return (
                cast(Callable[..., Any], getattr(module, function_name)),
                f"[bold]{name} enabled: {hook}[/bold]\n  {message}",
            )
        except (AttributeError, ModuleNotFoundError):
            raise PrerequisiteError(
                f"{env} provider not found: {hook}\n"
                + "Please correct (or undefine) this environment variable before proceeding."
            )
    else:
        return None
