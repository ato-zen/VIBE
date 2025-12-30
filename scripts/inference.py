"""Inference CLI."""

import json
import os
from pathlib import Path

import click
from loguru import logger
from PIL import Image

from src.editor import ImageEditor


def process_instruction(
    ctx: click.Context,  # noqa: ARG001
    param: click.Parameter,  # noqa: ARG001
    value: tuple[str, ...],
) -> str | list[str]:
    """Process instruction argument.

    Args:
        ctx (click.Context): The click context.
        param (click.Parameter): The click parameter.
        value (tuple[str, ...]): The value of the instruction argument.

    Returns:
        str | list[str]: The processed instruction.
    """
    if len(value) == 1:
        return value[0]
    return list(value)


@click.group()
def inference_cli() -> None:
    """Inference CLI."""
    return


@inference_cli.command()
@click.option("--instruction", type=str, required=True, multiple=True, callback=process_instruction)
@click.option("--height", type=int, default=1024)
@click.option("--width", type=int, default=1024)
@click.option("--num-images-per-prompt", type=int, default=1)
@click.option("--output-path", type=str, required=True)
@click.option("--checkpoint-path", type=str, required=True)
@click.option("--guidance-scale", type=float, default=4.5)
@click.option("--num-inference-steps", type=int, default=20)
@click.option("--device", type=str, default="cuda:0")
def generate_single_image(  # noqa: PLR0913
    instruction: str | list[str],
    height: int,
    width: int,
    num_images_per_prompt: int,
    output_path: str,
    checkpoint_path: str,
    guidance_scale: float,
    num_inference_steps: int,
    device: str,
) -> None:
    """Generate a single image.

    Args:
        instruction (str | list[str]): The generation prompt or list of prompts.
        height (int): The height of the image.
        width (int): The width of the image.
        num_images_per_prompt (int): The number of images to generate for each instruction.
        output_path (str): The path to the output images.
        checkpoint_path (str): The path to the local checkpoint.
        guidance_scale (float): Text guidance scale.
        num_inference_steps (int): The number of inference steps.
        device (str): The device to use.
    """
    os.makedirs(output_path, exist_ok=True)
    # Initialize the editor class
    editor = ImageEditor(
        checkpoint_path=checkpoint_path,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    # Generate edited images
    generated_images = editor.generate_edited_image(
        instruction,
        num_images_per_prompt=num_images_per_prompt,
        t2i_height=height,
        t2i_width=width,
    )

    # Save edited images
    for i, generated_image in enumerate(generated_images):
        generated_image.save(f"{output_path}/generated_image_{i}.jpg", quality=100)


@inference_cli.command()
@click.option("--image-path", type=str, required=True)
@click.option("--instruction", type=str, required=True, multiple=True, callback=process_instruction)
@click.option("--num-images-per-prompt", type=int, default=1)
@click.option("--output-path", type=str, required=True)
@click.option("--checkpoint-path", type=str, required=True)
@click.option("--image-guidance-scale", type=float, default=1.5)
@click.option("--guidance-scale", type=float, default=4.5)
@click.option("--num-inference-steps", type=int, default=20)
@click.option("--device", type=str, default="cuda:0")
def edit_single_image(  # noqa: PLR0913
    image_path: str,
    instruction: str | list[str],
    num_images_per_prompt: int,
    output_path: str,
    checkpoint_path: str,
    image_guidance_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    device: str,
) -> None:
    """Edit a single image.

    Args:
        image_path (str): The path to the input image.
        instruction (str | list[str]): The editing prompt or list of prompts.
        num_images_per_prompt (int): The number of images to generate for each instruction.
        output_path (str): The path to the output images.
        checkpoint_path (str): The path to the local checkpoint.
        image_guidance_scale (float): Image guidance scale. NOTE: -1.5 => not use img cfg, 1.5 => use
        guidance_scale (float): Text guidance scale.
        num_inference_steps (int): The number of inference steps.
        device (str): The device to use.
    """
    os.makedirs(output_path, exist_ok=True)
    # Initialize the editor class
    editor = ImageEditor(
        checkpoint_path=checkpoint_path,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    # Load input image
    image = Image.open(image_path)

    # Generate edited images
    edited_images = editor.generate_edited_image(
        instruction,
        conditioning_image=image,
        num_images_per_prompt=num_images_per_prompt,
    )

    # Save edited images
    for i, edited_image in enumerate(edited_images):
        edited_image.save(f"{output_path}/edited_image_{i}.jpg", quality=100)


@inference_cli.command()
@click.option("--mapping-path", type=str, required=True)
@click.option("--num-images-per-prompt", type=int, default=1)
@click.option("--output-path", type=str, required=True)
@click.option("--checkpoint-path", type=str, required=True)
@click.option("--image-guidance-scale", type=float, default=1.5)
@click.option("--guidance-scale", type=float, default=4.5)
@click.option("--num-inference-steps", type=int, default=20)
@click.option("--device", type=str, default="cuda:0")
def edit_multiple_images(  # noqa: PLR0913
    mapping_path: str,
    num_images_per_prompt: int,
    output_path: str,
    checkpoint_path: str,
    image_guidance_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    device: str,
) -> None:
    """Edit a set of images using a JSON mapping file.

    The mapping file should be a JSON list of objects, where each object contains "image_path" and "instruction".
    Example mapping.json:
    [
        {"image_path": "/path/to/image1.jpg", "instruction": "make it a painting"},
        {"image_path": "/path/to/image1.jpg", "instruction": "make it a sketch"},
        {"image_path": "/path/to/image2.png", "instruction": "add fireworks"},
        {"image_path": "/path/to/image3.png", "instruction": ["make it a painting", "make it a sketch"]}
    ]

    Args:
        mapping_path (str): The path to the JSON mapping file.
        num_images_per_prompt (int): The number of images to generate for each instruction.
        output_path (str): The path to the output directory.
        checkpoint_path (str): The path to the local checkpoint.
        image_guidance_scale (float): Image guidance scale. NOTE: -1.5 => not use img cfg, 1.5 => use
        guidance_scale (float): Text guidance scale.
        num_inference_steps (int): The number of inference steps.
        device (str): The device to use.
    """
    os.makedirs(output_path, exist_ok=True)
    # Initialize the editor class
    editor = ImageEditor(
        checkpoint_path=checkpoint_path,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    # Read mapping file
    with open(mapping_path, encoding="utf-8") as file:
        mapping = json.load(file)

    if not isinstance(mapping, list):
        msg = "Mapping file must be a JSON list."
        raise TypeError(msg)

    for batch_idx, item in enumerate(mapping):
        image_path = item.get("image_path")
        instruction = item.get("instruction")

        if not image_path or not instruction:
            logger.warning(f"Item at index {batch_idx} missing 'image_path' or 'instruction'. Skipping.")
            continue

        try:
            # Load input image
            image = Image.open(image_path)
        except FileNotFoundError:
            logger.warning(f"Image not found at {image_path}. Skipping.")
            continue

        # Generate edited images
        edited_images = editor.generate_edited_image(
            instruction,
            conditioning_image=image,
            num_images_per_prompt=num_images_per_prompt,
        )

        # Save edited images
        stem = Path(image_path).stem
        for i, edited_image in enumerate(edited_images):
            # Include batch_idx to ensure uniqueness if the same image is processed multiple times
            save_path = os.path.join(output_path, f"{stem}_idx{batch_idx}_{i}.jpg")
            edited_image.save(save_path, quality=100)


if __name__ == "__main__":
    inference_cli()
