# ------------------------------------------------------------------------------
# VIBE Gradio Interface
# Original VIBE Model by: AI-Forever / Alibaba / NVLabs
# Gradio Web UI Implementation by: ato-zen
# Repository: https://github.com/ato-zen/VIBE
# ------------------------------------------------------------------------------

import os
import sys
import torch
import random
import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from datetime import datetime
import inspect

# --- 1. CONFIGURATION ---
CACHE_DIR = "cache"
os.environ["CACHE_HOME"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR

sys.path.append(os.getcwd())

CHECKPOINT_PATH = "/home/user/VIBE/VIBE-Image-Edit"
SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. MODEL LOADING ---
print(f"🔄 Initialization... Loading model from: {CHECKPOINT_PATH}")

try:
    from vibe.editor import ImageEditor
except ImportError as e:
    print("❌ Error: 'vibe' module not found.")
    raise e

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    editor = ImageEditor(checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# --- 3. LOGIC ---

def get_image_dims(img):
    """Updates dimension fields when an image is uploaded."""
    if img is not None:
        return img.width, img.height, f"ℹ️ Original size: {img.width}x{img.height}"
    return 1024, 1024, "ℹ️ Text Mode (Default 1024x1024)"

def process_image(input_image, instruction, neg_prompt, img_guidance, text_guidance, steps, seed, randomize_seed, width, height):
    
    # Validation
    if input_image is None and not instruction:
        raise gr.Error("Please provide an Image OR an Instruction!")

    # Seed Handling
    if randomize_seed:
        seed = random.randint(0, 2147483647)
    
    # Set seed globally
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Target dimensions
    target_w = int(width)
    target_h = int(height)
    
    dims_info = f"{target_w}x{target_h}"
    print(f"🎨 Start: '{instruction}' | Seed: {seed} | Target Size: {dims_info}")

    try:
        # Determine method name
        target_method = editor.generate_edited_image
        
        # Prepare arguments
        kwargs = {
            "instruction": instruction,
            "num_inference_steps": steps,
            "guidance_scale": text_guidance,
            "image_guidance_scale": img_guidance,
            "seed": seed,
            "randomize_seed": False, 
            "num_images_per_prompt": 1,
            # We pass these just in case (for T2I mode)
            "t2i_width": target_w,
            "t2i_height": target_h,
        }

        # Handle Image Input & RESIZING LOGIC
        if input_image is not None:
            # Check if resize is needed
            if input_image.width != target_w or input_image.height != target_h:
                print(f"⚠️ Resizing input image from {input_image.size} to ({target_w}, {target_h})")
                # LANCZOS filter provides high quality downscaling/upscaling
                input_image = input_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
            kwargs["conditioning_image"] = input_image
        else:
            kwargs["conditioning_image"] = None

        # Handle Negative Prompt
        sig = inspect.signature(target_method)
        if "negative_prompt" in sig.parameters and neg_prompt:
            kwargs["negative_prompt"] = neg_prompt

        print(f"🚀 Running inference...")

        # Execute
        result = target_method(**kwargs)

        # Extract Result
        final_image = None
        if isinstance(result, Image.Image):
            final_image = result
        elif isinstance(result, list) and result:
            final_image = result[0]
        elif hasattr(result, "images") and result.images:
            final_image = result.images[0]
        
        if final_image is None:
            raise gr.Error("Error: Model returned no image.")

        # --- METADATA PREPARATION ---
        metadata = PngInfo()
        metadata.add_text("Prompt", str(instruction))
        if neg_prompt:
            metadata.add_text("Negative Prompt", str(neg_prompt))
        metadata.add_text("Seed", str(seed))
        metadata.add_text("Steps", str(steps))
        metadata.add_text("Guidance Scale", str(text_guidance))
        metadata.add_text("Image Guidance Scale", str(img_guidance))
        metadata.add_text("Target Size", f"{target_w}x{target_h}")
        metadata.add_text("Model", "VIBE")
        metadata.add_text("Interface", "ato-zen Gradio UI")

        # --- SAVING ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SAVE_DIR, f"{timestamp}.png")
        
        final_image.save(save_path, pnginfo=metadata)
        print(f"💾 Saved to: {save_path}")
        
        return final_image, seed

    except Exception as e:
        print(f"❌ Execution Error: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(str(e))


# --- 4. GRADIO INTERFACE ---

css = ""

with gr.Blocks(theme=gr.themes.Default(), title="VIBE Editor") as demo:
    
    gr.Markdown("## VIBE Editor (WebUI by ato-zen)")
    
    with gr.Row():
        # --- LEFT COLUMN (Inputs) ---
        with gr.Column(scale=4):
            
            # Input Image
            input_img = gr.Image(
                label="Input Image (Drag & Drop)", 
                type="pil", 
                sources=["upload", "clipboard"],
                interactive=True,
                height=400
            )
            
            res_info = gr.Markdown("ℹ️ Waiting for image...")

            prompt = gr.Textbox(
                label="Instruction (Prompt)", 
                placeholder="What to edit? (e.g. remove dog, make it winter)", 
                lines=3,
                autofocus=True
            )
            
            neg_prompt = gr.Textbox(
                label="Negative Prompt", 
                placeholder="What to avoid? (e.g. blur, text, distortion)", 
                visible=True
            )
            
            with gr.Row():
                width = gr.Number(value=1024, label="Width", precision=0)
                height = gr.Number(value=1024, label="Height", precision=0)

            with gr.Accordion("⚙️ Advanced Parameters", open=False):
                with gr.Row():
                    steps = gr.Slider(1, 100, value=20, step=1, label="Inference Steps")
                    seed = gr.Number(value=-1, label="Seed (-1 = Random)", precision=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                
                with gr.Row():
                    img_scale = gr.Slider(0.0, 5.0, value=1.2, step=0.1, label="Image Guidance Scale")
                    text_scale = gr.Slider(1.0, 20.0, value=4.5, step=0.5, label="Text Guidance Scale")

            btn_run = gr.Button("✨ GENERATE", variant="primary", size="lg")

        # --- RIGHT COLUMN (Outputs) ---
        with gr.Column(scale=5):
            output_img = gr.Image(label="Result", type="pil", interactive=False)
            with gr.Row():
                used_seed = gr.Number(label="Used Seed", interactive=False)

    # --- EVENTS ---
    
    # Auto-resize on upload
    input_img.upload(
        fn=get_image_dims,
        inputs=[input_img],
        outputs=[width, height, res_info]
    )

    # Run Button
    btn_run.click(
        fn=process_image,
        inputs=[input_img, prompt, neg_prompt, img_scale, text_scale, steps, seed, randomize_seed, width, height],
        outputs=[output_img, used_seed]
    )

if __name__ == "__main__":
    demo.queue().launch(share=False, allowed_paths=[SAVE_DIR])
