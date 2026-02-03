"""
SDXL-Turbo Image Generation Worker

Uses ModelRef injection for automatic model loading, caching, and lifecycle.
A custom runtime loader handles fp16 dtype and variant settings.
"""

import base64
from io import BytesIO
from typing import Annotated, Optional

import msgspec
import torch
from diffusers import AutoPipelineForText2Image
from gen_worker import ActionContext, worker_function
from gen_worker.injection import (
    ModelArtifacts,
    ModelRef,
    ModelRefSource as Src,
    register_runtime_loader,
)


def _load_sdxl_turbo_pipeline(
    ctx: ActionContext,
    artifacts: ModelArtifacts,
) -> AutoPipelineForText2Image:
    """Custom loader: ensures fp16 on GPU and correct variant."""
    model_id = artifacts.model_id

    # Strip the hf: prefix that _canonicalize_model_ref_string adds
    if model_id.startswith("hf:"):
        model_id = model_id[3:]

    device = str(ctx.device)
    is_gpu = device != "cpu"

    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if is_gpu else torch.float32,
        variant="fp16" if is_gpu else None,
    ).to(device)

    return pipeline


register_runtime_loader(AutoPipelineForText2Image, _load_sdxl_turbo_pipeline)


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
    pipeline: Annotated[
        AutoPipelineForText2Image,
        ModelRef(Src.DEPLOYMENT, "sdxl-turbo"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """Generate an image and save to file store."""
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


@worker_function()
def generate_base64(
    ctx: ActionContext,
    pipeline: Annotated[
        AutoPipelineForText2Image,
        ModelRef(Src.DEPLOYMENT, "sdxl-turbo"),
    ],
    payload: GenerateBase64Input,
) -> GenerateBase64Output:
    """Generate an image and return as base64 string."""
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
