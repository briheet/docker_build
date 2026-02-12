"""
SDXL-Turbo Image Generation Worker

Demonstrates proper gen-worker SDK usage with model injection.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Annotated, Optional

import msgspec
from diffusers import AutoPipelineForText2Image

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src
from gen_worker.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class GenerateInput(msgspec.Struct):
    """Input for the generate function."""

    prompt: str
    num_steps: int = 4
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    guidance_scale: float = 0.0  # SDXL-Turbo doesn't need guidance


class GenerateOutput(msgspec.Struct):
    """Output from the generate function."""

    image: Asset
    prompt: str
    settings: dict


class GenerateBase64Input(msgspec.Struct):
    """Input for the generate_base64 function."""

    prompt: str
    num_steps: int = 4
    width: int = 512
    height: int = 512
    seed: Optional[int] = None


class GenerateBase64Output(msgspec.Struct):
    """Output from the generate_base64 function."""

    image_base64: str
    prompt: str
    settings: dict


@worker_function(ResourceRequirements())
def generate(
    ctx: ActionContext,
    pipeline: Annotated[
        AutoPipelineForText2Image, ModelRef(Src.DEPLOYMENT, "sdxl_turbo")
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    Generate an image from a text prompt and save to file store.

    The pipeline is automatically injected by the worker runtime's model cache.
    This avoids global mutable state and enables proper model management.

    Args:
        ctx: Action context provided by the worker runtime
        pipeline: SDXL-Turbo pipeline, injected by the worker runtime
        payload: Input payload containing prompt and generation parameters

    Returns:
        GenerateOutput containing the Asset reference to the saved image
    """
    if ctx.is_canceled():
        raise InterruptedError("Task cancelled")

    logger.info(
        "[run_id=%s] sdxl-turbo prompt=%r steps=%s",
        ctx.run_id,
        payload.prompt,
        payload.num_steps,
    )

    # Set seed for reproducibility
    generator = None
    if payload.seed is not None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(payload.seed)

    # Generate image using injected pipeline
    image = pipeline(
        prompt=payload.prompt,
        num_inference_steps=payload.num_steps,
        guidance_scale=payload.guidance_scale,
        width=payload.width,
        height=payload.height,
        generator=generator,
    ).images[0]

    # Save image to file store using ctx
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    # ctx.save_bytes returns an Asset object
    asset = ctx.save_bytes(
        f"runs/{ctx.run_id}/outputs/image.png",
        buffer.getvalue(),
    )

    return GenerateOutput(
        image=asset,
        prompt=payload.prompt,
        settings={
            "num_steps": payload.num_steps,
            "width": payload.width,
            "height": payload.height,
            "seed": payload.seed,
            "guidance_scale": payload.guidance_scale,
        },
    )


@worker_function(ResourceRequirements())
def generate_base64(
    ctx: ActionContext,
    pipeline: Annotated[
        AutoPipelineForText2Image, ModelRef(Src.DEPLOYMENT, "sdxl_turbo")
    ],
    payload: GenerateBase64Input,
) -> GenerateBase64Output:
    """
    Generate an image and return as base64 string.

    Useful for API responses where direct file storage is not needed.

    Args:
        ctx: Action context provided by the worker runtime
        pipeline: SDXL-Turbo pipeline, injected by the worker runtime
        payload: Input payload containing prompt and generation parameters

    Returns:
        GenerateBase64Output containing the base64-encoded image
    """
    if ctx.is_canceled():
        raise InterruptedError("Task cancelled")

    logger.info(
        "[run_id=%s] sdxl-turbo (base64) prompt=%r steps=%s",
        ctx.run_id,
        payload.prompt,
        payload.num_steps,
    )

    # Set seed for reproducibility
    generator = None
    if payload.seed is not None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(payload.seed)

    # Generate image using injected pipeline
    image = pipeline(
        prompt=payload.prompt,
        num_inference_steps=payload.num_steps,
        guidance_scale=0.0,  # SDXL-Turbo optimal setting
        width=payload.width,
        height=payload.height,
        generator=generator,
    ).images[0]

    # Convert to base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return GenerateBase64Output(
        image_base64=img_base64,
        prompt=payload.prompt,
        settings={
            "num_steps": payload.num_steps,
            "width": payload.width,
            "height": payload.height,
            "seed": payload.seed,
        },
    )
