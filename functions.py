"""
SDXL-Turbo Image Generation Worker

TEMPORARY: Dummy responses to isolate framework vs function errors.
Includes monkey-patch to log exceptions from gen-worker's silent handler.
"""

import logging
import traceback as _tb
from typing import Optional

import msgspec
from gen_worker import worker_function, ActionContext

logger = logging.getLogger(__name__)

# --- Monkey-patch gen-worker to log exceptions in _execute_task ---
# The gen-worker silently swallows all exceptions without logging.
# This patch makes _map_exception also log the full traceback.
try:
    import gen_worker.worker as _gw

    _orig_map_exception = _gw.Worker._map_exception

    def _patched_map_exception(self, exc):
        tb_str = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
        logger.error("TASK EXCEPTION (patched): %s\n%s", exc, tb_str)
        return _orig_map_exception(self, exc)

    _gw.Worker._map_exception = _patched_map_exception
    logger.info("Monkey-patched Worker._map_exception for error logging")
except Exception as patch_err:
    logger.warning("Failed to monkey-patch Worker._map_exception: %s", patch_err)
# --- End monkey-patch ---


class GenerateInput(msgspec.Struct):
    prompt: str
    num_steps: int = 4
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    guidance_scale: float = 0.0


class GenerateOutput(msgspec.Struct):
    image_url: str
    prompt: str
    settings: dict


class GenerateBase64Input(msgspec.Struct):
    prompt: str
    num_steps: int = 4
    width: int = 512
    height: int = 512
    seed: Optional[int] = None


class GenerateBase64Output(msgspec.Struct):
    image_base64: str
    prompt: str
    settings: dict


@worker_function()
def generate(
    ctx: ActionContext,
    payload: GenerateInput,
) -> GenerateOutput:
    """Generate an image and save to file store (DUMMY for debugging)."""
    logger.info("generate called (DUMMY): prompt=%s", payload.prompt[:50])
    return GenerateOutput(
        image_url="https://example.com/dummy.png",
        prompt=payload.prompt,
        settings={
            "num_steps": payload.num_steps,
            "width": payload.width,
            "height": payload.height,
            "seed": payload.seed,
            "guidance_scale": payload.guidance_scale,
        },
    )


@worker_function()
def generate_base64(
    ctx: ActionContext,
    payload: GenerateBase64Input,
) -> GenerateBase64Output:
    """Generate an image as base64 (DUMMY for debugging)."""
    logger.info("generate_base64 called (DUMMY): prompt=%s", payload.prompt[:50])
    return GenerateBase64Output(
        image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        prompt=payload.prompt,
        settings={
            "num_steps": payload.num_steps,
            "width": payload.width,
            "height": payload.height,
            "seed": payload.seed,
        },
    )
