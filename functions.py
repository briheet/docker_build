"""
SDXL-Turbo Image Generation Worker

Demonstrates proper gen-worker SDK usage with model injection.

Works on:
- NVIDIA GPU (CUDA) - for RunPod deployment
- CPU - fallback (slow but works)
"""

from io import BytesIO
from typing import Annotated, Optional

import msgspec
import torch
from diffusers import AutoPipelineForText2Image
from gen_worker import worker_function, ActionContext, ModelRef, ModelRefSource


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
    image_url: str
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


@worker_function()
def generate(
    ctx: ActionContext,
    payload: GenerateInput,
    pipeline: Annotated[
        AutoPipelineForText2Image,
        ModelRef(ModelRefSource.DEPLOYMENT, "sdxl-turbo")
    ],
) -> GenerateOutput:
    """
    Generate an image from a text prompt and save to file store.

    The pipeline is automatically injected by the worker runtime's model cache.
    This avoids global mutable state and enables proper model management.

    Args:
        ctx: Action context provided by the worker runtime
        payload: Input payload containing prompt and generation parameters
        pipeline: SDXL-Turbo pipeline, injected by the worker runtime

    Returns:
        GenerateOutput containing the URL to the saved image
    """
    # Set seed for reproducibility
    generator = None
    if payload.seed is not None:
        generator = torch.Generator(device=ctx.device).manual_seed(payload.seed)

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
    buffer.seek(0)

    # Use ctx to save bytes and get URL
    image_url = ctx.save_bytes(
        f"generated/{ctx.run_id}.png",
        buffer.getvalue(),
        "image/png",
    )

    return GenerateOutput(
        image_url=image_url,
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
    pipeline: Annotated[
        AutoPipelineForText2Image,
        ModelRef(ModelRefSource.DEPLOYMENT, "sdxl-turbo")
    ],
) -> GenerateBase64Output:
    """
    Generate an image and return as base64 string.

    Useful for API responses where direct file storage is not needed.

    Args:
        ctx: Action context provided by the worker runtime
        payload: Input payload containing prompt and generation parameters
        pipeline: SDXL-Turbo pipeline, injected by the worker runtime

    Returns:
        GenerateBase64Output containing the base64-encoded image
    """
    import base64

    # Set seed for reproducibility
    generator = None
    if payload.seed is not None:
        generator = torch.Generator(device=ctx.device).manual_seed(payload.seed)

    # Generate image using injected pipeline
    image = pipeline(
        prompt=payload.prompt,
        num_inference_steps=payload.num_steps,
        guidance_scale=0.0,
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
