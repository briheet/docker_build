"""
SDXL-Turbo Image Generation Worker

Loads the pipeline directly (no ModelRef injection) for simple-mode compatibility.
"""

import base64
import logging
import traceback
from io import BytesIO
from typing import Optional

import msgspec
import torch
from diffusers import AutoPipelineForText2Image
from gen_worker import worker_function, ActionContext

logger = logging.getLogger(__name__)

# Global pipeline cache — loaded once on first call
_pipeline = None


def _get_pipeline(device: str):
    global _pipeline
    if _pipeline is None:
        logger.info("Loading SDXL-Turbo pipeline for device=%s...", device)
        _pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            variant="fp16" if device != "cpu" else None,
        ).to(device)
        logger.info("Pipeline loaded successfully.")
    return _pipeline


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
    """Generate an image and save to file store."""
    try:
        logger.info("generate called: prompt=%s, device=%s", payload.prompt[:50], ctx.device)
        pipeline = _get_pipeline(ctx.device)

        generator = None
        if payload.seed is not None:
            generator = torch.Generator(device=ctx.device).manual_seed(payload.seed)

        image = pipeline(
            prompt=payload.prompt,
            num_inference_steps=payload.num_steps,
            guidance_scale=payload.guidance_scale,
            width=payload.width,
            height=payload.height,
            generator=generator,
        ).images[0]

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

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
    except Exception as e:
        logger.error("generate FAILED: %s\n%s", e, traceback.format_exc())
        raise


@worker_function()
def generate_base64(
    ctx: ActionContext,
    payload: GenerateBase64Input,
) -> GenerateBase64Output:
    """Generate an image and return as base64 string."""
    try:
        logger.info("generate_base64 called: prompt=%s, device=%s", payload.prompt[:50], ctx.device)
        pipeline = _get_pipeline(ctx.device)

        generator = None
        if payload.seed is not None:
            generator = torch.Generator(device=ctx.device).manual_seed(payload.seed)

        image = pipeline(
            prompt=payload.prompt,
            num_inference_steps=payload.num_steps,
            guidance_scale=0.0,
            width=payload.width,
            height=payload.height,
            generator=generator,
        ).images[0]

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
    except Exception as e:
        logger.error("generate_base64 FAILED: %s\n%s", e, traceback.format_exc())
        raise
