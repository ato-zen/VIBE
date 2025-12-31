#!/bin/bash
#
# Usage:
#   chmod +x scripts/inference_template.sh
#   ./scripts/inference_template.sh

# set environment variables
export CACHE_HOME="path/to/your/cache/dir"  # <--- change it to your cache dir
export HF_HOME="$CACHE_HOME"
export TRANSFORMERS_CACHE="$CACHE_HOME"
export TORCH_HOME="$CACHE_HOME"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Set the checkpoint path
CHECKPOINT_PATH="path/to/checkpoint"  # <--- change it to your local CHECKPOINT_PATH

# Set the image path
IMAGE_PATH="examples/dog.jpg"

# Set the editing prompt
EDITING_PROMPT="turn the dog into a cat"

# Set the number of images per prompt
NUM_IMAGES_PER_PROMPT=1

# Set the output path
OUTPUT_PATH="examples/edited_images"

# Run the inference
python scripts/inference.py edit-single-image \
    --image-path "$IMAGE_PATH" \
    --instruction "$EDITING_PROMPT" \
    --num-images-per-prompt "$NUM_IMAGES_PER_PROMPT" \
    --output-path "$OUTPUT_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --image-guidance-scale 1.2 \
    --guidance-scale 4.5 \
    --num-inference-steps 20 \
    --device "cuda:0"
