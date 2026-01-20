import modal
import io
import base64
from pathlib import Path
from typing import Optional

# Create Modal app
app = modal.App("carpet-wizard")

# Default negative prompt for rug generation
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, warped, photographic, realistic, "
    "photograph, 3D, dimensional, shadows, lighting effects, depth, "
    "perspective, people, faces, text, watermark, signature, frame, "
    "border, poor quality, pixelated, noisy, grainy, artifacts, "
    "carpet texture, fabric texture, weave, fibers, physical material, "
    "photography, studio lighting, product shot, wildlife photography, "
    "nature documentary, safari, zoo, animals in wild, natural habitat, "
    "realistic animals, photorealistic wildlife, animal photography, "
    "nature scene, landscape, outdoor scene, real animals, living creatures, "
    "naturalistic, documentary style, animal portrait, wildlife scene, "
    "environmental background, forest scene, jungle scene, savanna, "
    "natural environment, realistic fur, realistic feathers, realistic scales, "
    "flowers, floral, petals, blooms, blossoms, roses, tulips, daisies, "
    "botanical, garden, meadow, field of flowers, flower heads, flower buds, "
    "flowering plants, flowering vines, floral patterns, floral motifs, "
    "flower arrangements, bouquets, corsages, garlands, wreaths, "
    "lily, orchid, sunflower, peony, carnation, iris, hibiscus, jasmine, "
    "cherry blossom, sakura, lotus flower, water lily, poppy, lavender, "
    "marigold, chrysanthemum, magnolia, azalea, camellia, gardenia, "
    "flower garden, botanical garden, greenhouse, nursery, floriculture, "
    "petal-like, flower-like, blossom-like, bloom-shaped, "
    "seams, visible edges, discontinuity, mismatched patterns, "
    "asymmetric, uneven, irregular spacing, broken pattern"
)

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "flask",
        "torch",
        "diffusers[torch]",
        "huggingface_hub",
        "accelerate",
        "transformers",
        "sentencepiece",
        "protobuf",
        "gradio_client",
        "torchvision"
    )
    .add_local_file(
        local_path=str(Path(__file__).parent / "rug-options.json"),
        remote_path="/app/rug-options.json"
    )
)

# Create volume for model caching
model_volume = modal.Volume.from_name("flux-model-cache", create_if_missing=True)

# Session and image storage
user_sessions = {}
generated_images = {}

# Step data (same as in app.py)
STEP_DATA = [
    {
        "id": 1,
        "title": "Where will this rug live?",
        "description": "Choose how the rug will be used. This helps determine durability, thickness, and knot density.",
        "keywords": ["Location", "Room", "Placement"],
        "options": [
            {"id": "living-room", "title": "Living Room", "desc": "High traffic, needs durability", "icon": "fa-couch"},
            {"id": "bedroom", "title": "Bedroom", "desc": "Comfort focused, lower traffic", "icon": "fa-bed"},
            {"id": "dining-area", "title": "Dining Area", "desc": "Spill resistant, easy to clean", "icon": "fa-utensils"},
            {"id": "office", "title": "Office or Studio", "desc": "Style meets function", "icon": "fa-briefcase"},
            {"id": "public-space", "title": "Hotel or Public Space", "desc": "Maximum durability", "icon": "fa-building"},
            {"id": "wall-art", "title": "Wall or Art Rug", "desc": "Decorative, less durable", "icon": "fa-image"}
        ]
    },
    {
        "id": 2,
        "title": "How big should it be?",
        "description": "Think about the space, not the design yet. Consider furniture layout and room proportions.",
        "keywords": ["Size", "Dimensions", "Scale"],
        "options": [
            {"id": "small", "title": "Small", "desc": "3'x5' to 5'x8'", "icon": "fa-expand-alt"},
            {"id": "medium", "title": "Medium", "desc": "6'x9' to 8'x10'", "icon": "fa-square"},
            {"id": "large", "title": "Large", "desc": "9'x12' and above", "icon": "fa-expand-arrows-alt"},
            {"id": "custom", "title": "Custom Size", "desc": "Specify exact dimensions", "icon": "fa-ruler-combined"},
            {"id": "runner", "title": "Runner", "desc": "Long and narrow", "icon": "fa-road"},
            {"id": "round", "title": "Round", "desc": "Circular shape", "icon": "fa-circle"}
        ]
    },
    {
        "id": 3,
        "title": "What mood do you want?",
        "description": "This sets the emotional direction of the rug. Choose the feeling you want the space to evoke.",
        "keywords": ["Mood", "Vibe", "Atmosphere"],
        "options": [
            {"id": "calm", "title": "Calm and Subtle", "desc": "Soothing, peaceful vibe", "icon": "fa-spa"},
            {"id": "bold", "title": "Bold and Expressive", "desc": "Eye-catching, statement piece", "icon": "fa-fire"},
            {"id": "warm", "title": "Warm and Cozy", "desc": "Inviting, comfortable feel", "icon": "fa-home"},
            {"id": "fresh", "title": "Fresh and Light", "desc": "Airy, bright atmosphere", "icon": "fa-sun"},
            {"id": "luxurious", "title": "Rich and Luxurious", "desc": "Opulent, premium look", "icon": "fa-gem"},
            {"id": "earthy", "title": "Earthy and Natural", "desc": "Organic, grounded feel", "icon": "fa-leaf"}
        ]
    },
    {
        "id": 4,
        "title": "Choose a style language",
        "description": "This defines the overall look of your rug. Select a style that matches your interior aesthetic.",
        "keywords": ["Style", "Aesthetic", "Design"],
        "options": [
            {"id": "traditional", "title": "Traditional", "desc": "Classic patterns and motifs", "icon": "fa-landmark"},
            {"id": "modern", "title": "Modern", "desc": "Clean lines, simple forms", "icon": "fa-cube"},
            {"id": "minimal", "title": "Minimal", "desc": "Essentials only, no clutter", "icon": "fa-minus"},
            {"id": "tribal", "title": "Tribal", "desc": "Ethnic, cultural patterns", "icon": "fa-globe-americas"},
            {"id": "contemporary", "title": "Contemporary", "desc": "Current trends, artistic", "icon": "fa-palette"},
            {"id": "experimental", "title": "Experimental", "desc": "Avant-garde, unconventional", "icon": "fa-flask"}
        ]
    },
    {
        "id": 5,
        "title": "Pick a color direction",
        "description": "Start broad, refine later. Choose a color family that fits your space and mood.",
        "keywords": ["Color", "Palette", "Hue"],
        "options": [
            {"id": "neutrals", "title": "Neutrals", "desc": "Beiges, grays, whites", "icon": "fa-circle"},
            {"id": "warm-tones", "title": "Warm Tones", "desc": "Reds, oranges, yellows", "icon": "fa-fire"},
            {"id": "cool-tones", "title": "Cool Tones", "desc": "Blues, greens, purples", "icon": "fa-snowflake"},
            {"id": "monochrome", "title": "Monochrome", "desc": "Single color palette", "icon": "fa-adjust"},
            {"id": "high-contrast", "title": "High Contrast", "desc": "Bold light/dark combos", "icon": "fa-star"},
            {"id": "soft-muted", "title": "Soft and Muted", "desc": "Pastels, subdued tones", "icon": "fa-cloud"}
        ]
    },
    {
        "id": 6,
        "title": "Generate Design",
        "description": "Generate your custom rug design based on all selections.",
        "keywords": ["Generate", "Create", "Design"]
    }
]


# FLUX-adapted seamless generation functions
def asymmetricConv2DConvForward_circular(self, input, weight, bias):
    """Circular padding for Conv2d layers - adapted for FLUX architecture"""
    import torch
    from torch.nn import functional as F
    from torch.nn.modules.utils import _pair

    self.paddingX = (
        self._reversed_padding_repeated_twice[0],
        self._reversed_padding_repeated_twice[1],
        0,
        0
    )

    self.paddingY = (
        0,
        0,
        self._reversed_padding_repeated_twice[2],
        self._reversed_padding_repeated_twice[3]
    )
    working = F.pad(input, self.paddingX, mode="circular")
    working = F.pad(working, self.paddingY, mode="circular")

    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


def make_seamless_flux(model):
    """Enable circular padding on all Conv2d layers in FLUX model"""
    import torch
    from torch.nn import Conv2d

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            if hasattr(module, 'lora_layer') and module.lora_layer is None:
                module.lora_layer = lambda *x: 0
            module._conv_forward = asymmetricConv2DConvForward_circular.__get__(module, Conv2d)


def disable_seamless_flux(model):
    """Disable circular padding and restore default Conv2d behavior"""
    import torch
    import torch.nn as nn
    from torch.nn import Conv2d

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            if hasattr(module, 'lora_layer') and module.lora_layer is None:
                module.lora_layer = lambda *x: 0
            module._conv_forward = nn.Conv2d._conv_forward.__get__(module, Conv2d)


@app.cls(
    image=image,
    gpu="H100",
    timeout=600,
    volumes={"/cache": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class FluxModel:
    @modal.enter()
    def load_model(self):
        """Load the FLUX.1-schnell model on container startup with H100 optimizations"""
        import torch
        from diffusers import FluxPipeline
        import os
        import gc

        # Set cache directory and optimized CUDA memory allocation for H100
        os.environ["HF_HOME"] = "/cache"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"

        # Enable TF32 for faster matmul on H100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        repo_id = "black-forest-labs/FLUX.1-schnell"
        device = "cuda:0"
        torch_dtype = torch.bfloat16

        # Clear any existing CUDA cache
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        print(f"Loading FLUX.1-schnell pipeline on {device}...")

        self.pipe = FluxPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            cache_dir="/cache",
            use_safetensors=True
        )

        # Enable attention slicing for memory efficiency
        self.pipe.enable_attention_slicing()

        # Enable CPU offload to save VRAM
        self.pipe.enable_model_cpu_offload()

        self.device = device

        gc.collect()
        torch.cuda.empty_cache()

        print(f"FLUX.1-schnell model loaded. Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    def flux_diffusion_callback(self, pipe, step_index, timestep, callback_kwargs):
        """
        Callback for seamless pattern generation adapted for FLUX architecture.
        FLUX uses rectified flow, so we adapt the timing and approach.
        """
        import torch

        # For FLUX, we apply seamless techniques in the last 20% of steps
        total_steps = pipe.num_inference_steps if hasattr(pipe, 'num_inference_steps') else 28
        late_stage_threshold = int(total_steps * 0.8)

        # Apply circular padding to transformer and VAE in late stages
        if step_index >= late_stage_threshold:
            if hasattr(pipe, 'transformer') and pipe.transformer is not None:
                make_seamless_flux(pipe.transformer)
            if hasattr(pipe, 'vae') and pipe.vae is not None:
                make_seamless_flux(pipe.vae)

        # Noise rolling for early stages (adapted for FLUX's flow matching)
        if step_index < late_stage_threshold:
            if "latents" in callback_kwargs:
                latents = callback_kwargs["latents"]
                if len(latents.shape) >= 4:
                    shift_amount = min(16, max(4, latents.shape[-1] // 32))
                    callback_kwargs["latents"] = torch.roll(
                        latents,
                        shifts=(shift_amount, shift_amount),
                        dims=(-2, -1)
                    )

        return callback_kwargs

    @modal.method()
    def generate_image(self, prompt: str, width: int = 1024, height: int = 1024, seed: int = 42, enable_seamless: bool = True):
        """Generate an image from a prompt with H100 optimizations and seamless pattern techniques"""
        import torch
        from io import BytesIO
        import gc

        try:
            print(f"Generating image with prompt:\n{prompt}")
            print(f"Negative prompt:\n{DEFAULT_NEGATIVE_PROMPT}")
            print(f"Image dimensions: {width}x{height}")
            print(f"Seamless mode: {enable_seamless}")
            print(f"GPU memory before generation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            # Clear cache before generation
            gc.collect()
            torch.cuda.empty_cache()

            # Ensure seamless is disabled before starting
            if enable_seamless:
                if hasattr(self.pipe, 'transformer') and self.pipe.transformer is not None:
                    disable_seamless_flux(self.pipe.transformer)
                if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
                    disable_seamless_flux(self.pipe.vae)

            # Create generator on CUDA for seamless generation
            generator = torch.Generator(device="cuda").manual_seed(seed)

            # Generate image with FLUX.1-schnell and seamless techniques
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if enable_seamless:
                    image = self.pipe(
                        prompt=prompt,
                        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                        width=width,
                        height=height,
                        generator=generator,
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        max_sequence_length=256,
                        callback_on_step_end=self.flux_diffusion_callback
                    ).images[0]
                else:
                    image = self.pipe(
                        prompt=prompt,
                        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                        width=width,
                        height=height,
                        generator=generator,
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        max_sequence_length=256,
                    ).images[0]

            # Cleanup after generation
            gc.collect()
            torch.cuda.empty_cache()

            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            img_str = base64.b64encode(img_bytes).decode()

            print(f"Image generated successfully. Size: {len(img_bytes)} bytes")

            return {
                "success": True,
                "image_data": img_str,
                "prompt": prompt
            }

        except Exception as e:
            print(f"Error generating image: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            return {"error": str(e)}


def get_dimensions_from_selection(size_selection: str, shape_selection: str = '') -> tuple[int, int]:
    """Map size and shape selection to image dimensions (width, height). All dimensions must be divisible by 16."""
    size_selection_lower = (size_selection or '').lower()
    shape_selection_lower = (shape_selection or '').lower()

    # Handle shape-based dimensions first
    if 'runner' in shape_selection_lower:
        return (1024, 512)
    if 'round' in shape_selection_lower or 'square' in shape_selection_lower:
        return (1024, 1024)

    # Handle size-based dimensions
    if '900' in size_selection or '1200' in size_selection:
        return (896, 1200)  # 900x1200 adjusted to be divisible by 16
    if 'small' in size_selection_lower or '120' in size_selection:
        return (768, 768)
    if 'large' in size_selection_lower or '340' in size_selection or '290' in size_selection:
        return (1024, 1024)

    # Default medium size
    return (1024, 1024)


def generate_prompt_from_selections(selections):
    """Convert user selections into a detailed prompt for FLUX.1-schnell

    Step order matches rug-options.json accordions:
    - step1 = rooms
    - step2 = design-style
    - step3 = size
    - step4 = shape
    - step5 = design-details
    - step6 = color
    - step7 = mood
    """

    # Step 1: Room/Location
    room = selections.get('step1', 'indoor space')

    # Step 2: Design Style
    style = selections.get('step2', 'custom')

    # Step 3: Size (used for image dimensions, not in prompt)

    # Step 4: Shape
    shape = selections.get('step4', 'Rectangle').lower()

    # Step 5: Design Details
    detail = selections.get('step5', '')

    # Step 6: Color
    color = selections.get('step6', '')

    # Step 7: Mood
    mood = selections.get('step7', '')

    # Build the main prompt structure
    prompt = f"Flat 2D vector illustration of {style} rug design suitable for a {room}"

    if shape:
        prompt += f" in {shape} shape"

    if detail:
        prompt += f", {detail}"

    if mood:
        prompt += f", {mood} mood"

    if color:
        prompt += f", {color} color palette"

    prompt += """.

Seamless repeating pattern, perfectly tileable,
exact pattern repeat, edge-to-edge continuity,
no framing, no scale indicators.

Strict vector artwork:
solid flat color fills only,
hard edges, sharp geometry,

NO texture of any kind:
no fabric texture, no grain, no noise,
no shading, no lighting,
no realism, no depth, no shadows, no 3D effects.

SVG / AI / EPS style,
screen-print ready, textile CAD pattern,
manufacturing-ready vector artwork.
"""

    return prompt.strip()



# HTML content embedded directly in deployment file (RugWise Studio UI)
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RugWise | Custom Rug Design Studio</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        :root {
            --primary: #7c3aed;
            --primary-light: #8b5cf6;
            --primary-dark: #6d28d9;
            --secondary: #10b981;
            --dark: #1f2937;
            --light: #f9fafb;
            --gray: #6b7280;
            --gray-light: #f3f4f6;
            --border: #e5e7eb;
            --shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            --radius: 10px;
            --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        body {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            color: var(--dark);
            min-height: 100vh;
            padding: 16px;
            font-size: 14px;
        }

        .container {
            display: flex;
            max-width: 1600px;
            margin: 0 auto;
            gap: 20px;
            height: calc(100vh - 32px);
            overflow: hidden;
        }

        /* Left Panel - Configuration */
        .config-panel {
            flex: 0 0 360px;
            background: white;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .panel-header {
            padding: 18px 20px;
            background: linear-gradient(to right, var(--primary), var(--primary-light));
            color: white;
            text-align: center;
        }

        .panel-header h1 {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 3px;
            letter-spacing: -0.2px;
        }

        .panel-header p {
            font-size: 12px;
            opacity: 0.85;
            font-weight: 400;
        }

        .config-content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }

        .step-info {
            display: flex;
            flex-direction: column;
        }

        .step-counter {
            font-size: 12px;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 2px;
        }

        .step-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--dark);
        }

        .step-navigation {
            display: flex;
            gap: 8px;
        }

        .nav-btn {
            width: 32px;
            height: 32px;
            border-radius: 6px;
            background: white;
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: var(--transition);
            color: var(--gray);
            font-size: 12px;
        }

        .nav-btn:hover:not(:disabled) {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        .nav-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .accordion-section {
            margin-bottom: 16px;
        }

        .accordion-header {
            background: var(--gray-light);
            padding: 14px 16px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            transition: var(--transition);
            border: 1px solid transparent;
            font-size: 13px;
        }

        .accordion-header:hover {
            background: #e5e7eb;
        }

        .accordion-header.active {
            background: white;
            border-color: var(--border);
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        }

        .accordion-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
            color: var(--dark);
        }

        .accordion-title i {
            font-size: 12px;
            width: 16px;
            text-align: center;
        }

        .accordion-icon {
            font-size: 10px;
            transition: var(--transition);
            color: var(--gray);
        }

        .accordion-header.active .accordion-icon {
            transform: rotate(180deg);
            color: var(--primary);
        }

        .accordion-content {
            max-height: 0;
            overflow: hidden;
            transition: var(--transition);
            padding: 0 16px;
        }

        .accordion-content.show {
            max-height: 280px;
            padding: 16px;
            overflow-y: auto;
        }

        .option-list {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }

        .option-item {
            position: relative;
        }

        .option-item input {
            position: absolute;
            opacity: 0;
            width: 0;
            height: 0;
        }

        .option-label {
            display: block;
            padding: 10px 8px;
            background: white;
            border: 1.5px solid var(--border);
            border-radius: 6px;
            text-align: center;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .option-item input:checked+.option-label {
            border-color: var(--primary);
            background: rgba(124, 58, 237, 0.05);
            color: var(--primary);
            font-weight: 600;
            box-shadow: 0 1px 4px rgba(124, 58, 237, 0.08);
        }

        .option-label:hover {
            border-color: var(--primary-light);
        }

        .custom-input {
            margin-top: 16px;
            grid-column: span 2;
        }

        .custom-input label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 12px;
            color: var(--dark);
        }

        .custom-input textarea {
            width: 100%;
            padding: 10px;
            border: 1.5px solid var(--border);
            border-radius: 6px;
            font-size: 12px;
            resize: vertical;
            transition: var(--transition);
            background: white;
            min-height: 60px;
        }

        .custom-input textarea:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.08);
        }

        .generate-btn-container {
            padding: 20px;
            background: white;
            border-top: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .generate-btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(to right, var(--primary), var(--primary-light));
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 2px 8px rgba(124, 58, 237, 0.2);
        }

        .generate-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
        }

        .generate-btn:active {
            transform: translateY(0);
        }

        .generate-btn:disabled {
            background: var(--gray);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .nav-buttons-container {
            display: flex;
            gap: 10px;
        }

        .nav-buttons-container button {
            flex: 1;
            padding: 10px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: var(--transition);
            border: 1px solid var(--border);
            background: white;
            color: var(--dark);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }

        .nav-buttons-container button:hover:not(:disabled) {
            background: var(--gray-light);
        }

        .nav-buttons-container button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .nav-buttons-container .prev-btn {
            border-color: var(--border);
        }

        .nav-buttons-container .next-btn {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }

        .nav-buttons-container .next-btn:hover:not(:disabled) {
            background: var(--primary-dark);
        }

        .preview-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 20px;
            min-width: 0;
        }

        .image-container {
            flex: 1;
            background: white;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .image-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .image-header h2 {
            font-size: 16px;
            font-weight: 700;
            color: var(--dark);
        }

        .image-status {
            font-size: 12px;
            color: var(--gray);
            font-weight: 500;
        }

        .image-placeholder {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 30px;
            position: relative;
            overflow: hidden;
        }

        .placeholder-content {
            text-align: center;
            max-width: 320px;
        }

        .placeholder-icon {
            font-size: 48px;
            color: var(--gray-light);
            margin-bottom: 16px;
        }

        .placeholder-content h3 {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--dark);
        }

        .placeholder-content p {
            color: var(--gray);
            margin-bottom: 16px;
            font-size: 13px;
            line-height: 1.5;
        }

        .hint {
            font-size: 12px;
            color: var(--primary);
            background: rgba(124, 58, 237, 0.05);
            padding: 8px 12px;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .image-display {
            display: none;
            width: 100%;
            height: 100%;
            position: relative;
        }

        .image-display.show {
            display: block;
        }

        .generated-image {
            width: 100%;
            height: 100%;
            object-fit: contain;
            padding: 20px;
        }

        .image-actions {
            position: absolute;
            bottom: 16px;
            right: 16px;
            display: flex;
            gap: 8px;
        }

        .image-action-btn {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: white;
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: var(--transition);
            color: var(--dark);
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
            font-size: 12px;
        }

        .image-action-btn:hover {
            background: var(--primary);
            color: white;
            transform: translateY(-1px);
        }

        .choices-panel {
            flex: 0 0 360px;
            background: white;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .choices-header {
            padding: 18px 20px;
            background: linear-gradient(to right, #0f766e, var(--secondary));
            color: white;
            text-align: center;
        }

        .choices-header h2 {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 3px;
            letter-spacing: -0.2px;
        }

        .choices-header p {
            font-size: 12px;
            opacity: 0.85;
            font-weight: 400;
        }

        .choices-content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }

        .choices-list {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .choice-item {
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }

        .choice-item:last-child {
            border-bottom: none;
        }

        .choice-label {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--dark);
            font-size: 13px;
        }

        .choice-icon {
            width: 28px;
            height: 28px;
            border-radius: 6px;
            background: rgba(124, 58, 237, 0.08);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--primary);
            font-size: 12px;
        }

        .choice-value {
            padding-left: 36px;
            color: var(--dark);
            font-size: 13px;
            line-height: 1.4;
            min-height: 18px;
        }

        .empty-choice {
            color: var(--gray);
            font-style: italic;
            font-size: 12px;
        }

        .json-viewer {
            margin-top: 20px;
            background: var(--gray-light);
            border-radius: 8px;
            padding: 16px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 11px;
            line-height: 1.4;
            max-height: 180px;
            overflow-y: auto;
            display: none;
        }

        .json-viewer.show {
            display: block;
        }

        .json-toggle {
            text-align: center;
            margin-top: 16px;
        }

        .json-toggle-btn {
            background: none;
            border: none;
            color: var(--primary);
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
            margin: 0 auto;
            padding: 6px 10px;
            border-radius: 4px;
            transition: var(--transition);
        }

        .json-toggle-btn:hover {
            background: rgba(124, 58, 237, 0.05);
        }

        .loading-spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: inline-block;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }

        @media (max-width: 1200px) {
            .container {
                flex-direction: column;
                height: auto;
                overflow: visible;
            }

            .config-panel, .choices-panel {
                flex: none;
                width: 100%;
            }

            .preview-panel {
                min-height: 400px;
            }
        }

        @media (max-width: 768px) {
            .option-list {
                grid-template-columns: 1fr;
            }

            .custom-input {
                grid-column: span 1;
            }

            body {
                padding: 12px;
                font-size: 13px;
            }

            .nav-buttons-container {
                flex-direction: column;
            }
        }
    </style>
</head>

<body>
    <div class="container">
        <!-- Left Panel - Configuration -->
        <div class="config-panel">
            <div class="panel-header">
                <h1>RugWise Studio</h1>
                <p>Design your perfect rug in simple steps</p>
            </div>

            <div class="config-content">
                <div id="accordions-container">
                    <!-- Accordions will be loaded from JSON -->
                </div>
            </div>

            <div class="generate-btn-container">
                <div class="nav-buttons-container">
                    <button class="prev-btn" id="prev-btn">
                        <i class="fas fa-arrow-left"></i>
                        Previous
                    </button>
                    <button class="next-btn" id="next-btn">
                        Next
                        <i class="fas fa-arrow-right"></i>
                    </button>
                </div>

                <button class="generate-btn" id="generate-btn">
                    <i class="fas fa-magic"></i>
                    Generate Rug Design
                </button>
            </div>
        </div>

        <!-- Center Panel - Image Preview -->
        <div class="preview-panel">
            <div class="image-container">
                <div class="image-header">
                    <h2>Design Preview</h2>
                    <div class="image-status" id="image-status">Ready to generate</div>
                </div>

                <div class="image-placeholder" id="image-placeholder">
                    <div class="placeholder-content">
                        <div class="placeholder-icon">
                            <i class="fas fa-th-large"></i>
                        </div>
                        <h3>Your Rug Awaits</h3>
                        <p>Configure your design on the left panel, then click "Generate Rug Design" to visualize your
                            creation.</p>
                        <div class="hint">
                            <i class="fas fa-lightbulb"></i> Use navigation buttons to move between steps
                        </div>
                    </div>
                </div>

                <div class="image-display" id="image-display">
                    <img class="generated-image" id="generated-image" src="" alt="Generated Rug Design">
                    <div class="image-actions">
                        <button class="image-action-btn" id="download-btn" title="Download Design">
                            <i class="fas fa-download"></i>
                        </button>
                        <button class="image-action-btn" id="refresh-btn" title="Regenerate">
                            <i class="fas fa-redo"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right Panel - Choices Preview -->
        <div class="choices-panel">
            <div class="choices-header">
                <h2>Your Selections</h2>
                <p>Design specifications</p>
            </div>

            <div class="choices-content">
                <div class="choices-list" id="choices-list">
                </div>

                <button class="json-toggle-btn" id="json-toggle-btn">
                    <i class="fas fa-code"></i>
                    Show JSON Data
                </button>

                <div class="json-viewer" id="json-viewer"></div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let stepData = [];
        let userChoices = {};
        let sessionId = '';
        let currentStep = 0;

        // Initialize the application
        document.addEventListener('DOMContentLoaded', async function () {
            sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
            console.log('Session ID:', sessionId);

            await loadStepData();
            await loadSelections();
            createAccordionFromConfig();
            createChoicesDisplay();
            updateChoicesDisplay();
            updateNavigationButtons();

            document.getElementById('generate-btn').addEventListener('click', generateDesign);
            document.getElementById('prev-btn').addEventListener('click', () => navigateToStep(currentStep - 1));
            document.getElementById('next-btn').addEventListener('click', () => navigateToStep(currentStep + 1));
            document.getElementById('json-toggle-btn').addEventListener('click', toggleJsonViewer);
            document.getElementById('download-btn').addEventListener('click', downloadGeneratedImage);
            document.getElementById('refresh-btn').addEventListener('click', generateDesign);
        });

        // Load step data from JSON file
        async function loadStepData() {
            try {
                const response = await fetch('rug-options.json');
                const data = await response.json();

                if (!data || !data.accordions || data.accordions.length === 0) {
                    throw new Error('JSON file is empty or invalid');
                }

                stepData = data.accordions;
                console.log('Loaded step data:', stepData);
            } catch (error) {
                console.error('Error loading JSON:', error);
                stepData = [];
            }
        }

        async function loadSelections() {
            try {
                const response = await fetch(`/api/get-selections/${sessionId}`);
                const data = await response.json();
                Object.keys(data).forEach(key => {
                    const stepNum = parseInt(key.replace('step', ''));
                    if (stepData[stepNum - 1]) {
                        userChoices[stepData[stepNum - 1].id] = data[key];
                    }
                });
            } catch (error) {
                console.error('Error loading selections:', error);
                userChoices = {};
            }
        }

        function createAccordionFromConfig() {
            const container = document.getElementById('accordions-container');
            container.innerHTML = '';

            stepData.forEach((accordion, index) => {
                const accordionElement = document.createElement('div');
                accordionElement.className = 'accordion-section';
                accordionElement.id = `accordion-${accordion.id}`;

                const contentId = `${accordion.id}-content`;
                const isOpen = index === currentStep || accordion.initiallyOpen;

                accordionElement.innerHTML = `
                    <div class="accordion-header ${isOpen ? 'active' : ''} ${userChoices[accordion.id] ? 'completed' : ''}" data-accordion="${accordion.id}" data-index="${index}">
                        <div class="accordion-title">
                            <i class="${accordion.icon}"></i>
                            <span>${accordion.title}</span>
                        </div>
                        <div class="accordion-icon">
                            <i class="fas fa-chevron-down"></i>
                        </div>
                    </div>
                    <div class="accordion-content ${isOpen ? 'show' : ''}" id="${contentId}">
                        <div class="option-list" id="${accordion.id}-options"></div>
                    </div>
                `;

                container.appendChild(accordionElement);

                const optionsContainer = document.getElementById(`${accordion.id}-options`);

                accordion.options.forEach((option, optionIndex) => {
                    const optionId = `${accordion.id}-option-${optionIndex}`;
                    const optionElement = document.createElement('div');
                    optionElement.className = 'option-item';

                    optionElement.innerHTML = `
                        <input type="${accordion.type}" id="${optionId}" name="${accordion.id}" value="${option}" ${userChoices[accordion.id] === option ? 'checked' : ''}>
                        <label for="${optionId}" class="option-label">${option}</label>
                    `;

                    optionsContainer.appendChild(optionElement);

                    const radioBtn = optionElement.querySelector('input');
                    radioBtn.addEventListener('change', function () {
                        updateUserChoice(accordion.id, option, index + 1);
                        updateChoicesDisplay();
                        updateNavigationButtons();
                    });
                });

                if (accordion.hasCustomInput) {
                    const customInputDiv = document.createElement('div');
                    customInputDiv.className = 'custom-input';
                    customInputDiv.innerHTML = `
                        <label for="${accordion.customInputId}">${accordion.customInputLabel}</label>
                        <textarea id="${accordion.customInputId}" placeholder="${accordion.customInputPlaceholder}" rows="2"></textarea>
                    `;
                    optionsContainer.appendChild(customInputDiv);

                    const textarea = document.getElementById(accordion.customInputId);
                    textarea.addEventListener('input', function () {
                        if (textarea.value.trim()) {
                            updateUserChoice(accordion.id, textarea.value.trim(), index + 1);
                            updateChoicesDisplay();
                            updateNavigationButtons();
                        }
                    });
                }
            });

            document.querySelectorAll('.accordion-header').forEach(header => {
                header.addEventListener('click', function () {
                    const index = parseInt(this.getAttribute('data-index'));
                    navigateToStep(index);
                });
            });
        }

        function updateUserChoice(category, value, stepNum) {
            userChoices[category] = value;
            
            fetch('/api/save-selection', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    step: stepNum,
                    selection: value
                })
            }).catch(err => console.error('Error saving:', err));

            const header = document.querySelector(`#accordion-${category} .accordion-header`);
            if (header) {
                header.classList.add('completed');
            }
        }

        function createChoicesDisplay() {
            const choicesList = document.getElementById('choices-list');
            choicesList.innerHTML = '';

            stepData.forEach(accordion => {
                const choiceItem = document.createElement('div');
                choiceItem.className = 'choice-item';
                choiceItem.id = `choice-${accordion.id}`;

                choiceItem.innerHTML = `
                    <div class="choice-label">
                        <div class="choice-icon">
                            <i class="${accordion.icon}"></i>
                        </div>
                        <span>${accordion.title}</span>
                    </div>
                    <div class="choice-value empty-choice">Not selected</div>
                `;

                choicesList.appendChild(choiceItem);
            });
        }

        function updateChoicesDisplay() {
            stepData.forEach(accordion => {
                const choiceItem = document.getElementById(`choice-${accordion.id}`);
                const valueElement = choiceItem.querySelector('.choice-value');
                
                if (userChoices[accordion.id]) {
                    valueElement.textContent = userChoices[accordion.id];
                    valueElement.classList.remove('empty-choice');
                } else {
                    valueElement.textContent = 'Not selected';
                    valueElement.classList.add('empty-choice');
                }
            });

            const jsonViewer = document.getElementById('json-viewer');
            jsonViewer.textContent = JSON.stringify(userChoices, null, 2);
        }

        function navigateToStep(index) {
            if (index < 0 || index >= stepData.length) return;

            document.querySelectorAll('.accordion-header').forEach((header, i) => {
                header.classList.remove('active');
                if (i === index) header.classList.add('active');
            });

            document.querySelectorAll('.accordion-content').forEach((content, i) => {
                content.classList.remove('show');
                if (i === index) content.classList.add('show');
            });

            currentStep = index;
            updateNavigationButtons();
        }

        function updateNavigationButtons() {
            const prevBtn = document.getElementById('prev-btn');
            const nextBtn = document.getElementById('next-btn');
            const generateBtn = document.getElementById('generate-btn');

            prevBtn.disabled = currentStep === 0;
            nextBtn.disabled = currentStep === stepData.length - 1;

            const allSelected = stepData.every(step => userChoices[step.id]);
            generateBtn.disabled = !allSelected;
        }

        function toggleJsonViewer() {
            const jsonViewer = document.getElementById('json-viewer');
            const toggleBtn = document.getElementById('json-toggle-btn');
            
            jsonViewer.classList.toggle('show');
            toggleBtn.innerHTML = jsonViewer.classList.contains('show') 
                ? '<i class="fas fa-code"></i> Hide JSON Data'
                : '<i class="fas fa-code"></i> Show JSON Data';
        }

        async function generateDesign() {
            const generateBtn = document.getElementById('generate-btn');
            const imagePlaceholder = document.getElementById('image-placeholder');
            const imageDisplay = document.getElementById('image-display');
            const imageStatus = document.getElementById('image-status');
            const generatedImage = document.getElementById('generated-image');

            const originalText = generateBtn.innerHTML;
            generateBtn.innerHTML = '<div class="loading-spinner"></div> Generating...';
            generateBtn.disabled = true;
            imageStatus.textContent = 'Generating your design...';

            try {
                const selections = {};
                stepData.forEach((step, index) => {
                    selections[`step${index + 1}`] = userChoices[step.id];
                });

                const response = await fetch('/api/generate-design', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        selections: selections
                    })
                });

                const data = await response.json();

                if (data.error) {
                    alert(`Error: ${data.error}`);
                    imageStatus.textContent = 'Generation failed';
                } else if (data.image_data) {
                    generatedImage.src = `data:image/png;base64,${data.image_data}`;
                    imagePlaceholder.style.display = 'none';
                    imageDisplay.classList.add('show');
                    imageStatus.textContent = 'Design complete!';
                }

            } catch (error) {
                console.error('Error:', error);
                alert('Failed to generate design.');
                imageStatus.textContent = 'Generation failed';
            }

            generateBtn.innerHTML = originalText;
            generateBtn.disabled = false;
        }

        async function downloadGeneratedImage() {
            try {
                const response = await fetch(`/api/download-image/${sessionId}`);
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `rug-design-${sessionId}.png`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                } else {
                    alert('Failed to download image');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Failed to download image.');
            }
        }
    </script>
</body>

</html>"""

# Web endpoints using Modal's WSGI support
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.wsgi_app()
def web_app():
    from flask import Flask, jsonify, request, Response
    import uuid

    flask_app = Flask(__name__)

    @flask_app.route('/')
    def index():
        return HTML_CONTENT

    @flask_app.route('/rug-options.json')
    def rug_options():
        import json
        try:
            with open('/app/rug-options.json', 'r') as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route('/api/steps')
    def get_steps():
        return jsonify(STEP_DATA)

    @flask_app.route('/api/step/<int:step_id>')
    def get_step(step_id):
        if 1 <= step_id <= len(STEP_DATA):
            return jsonify(STEP_DATA[step_id - 1])
        return jsonify({"error": "Step not found"}), 404

    @flask_app.route('/api/save-selection', methods=['POST'])
    def save_selection():
        data = request.json
        session_id = data.get('session_id', 'default')
        step = data.get('step')
        selection = data.get('selection')

        if session_id not in user_sessions:
            user_sessions[session_id] = {}

        user_sessions[session_id][f'step{step}'] = selection

        return jsonify({"success": True, "session_id": session_id})

    @flask_app.route('/api/get-selections/<session_id>')
    def get_selections(session_id):
        if session_id in user_sessions:
            return jsonify(user_sessions[session_id])
        return jsonify({})

    @flask_app.route('/api/generate-design', methods=['POST'])
    def generate_design():
        data = request.json
        session_id = data.get('session_id', str(uuid.uuid4()))

        # Try to get selections from request body first, then fall back to server session
        selections = data.get('selections')

        if not selections:
            # Fallback to server-side session storage
            if session_id not in user_sessions:
                return jsonify({"error": "No selections found for this session"}), 400
            selections = user_sessions[session_id]
        else:
            # Store selections for this session
            user_sessions[session_id] = selections

        # Check if we have enough selections (7 steps in JSON)
        if len(selections) < 7:
            return jsonify({"error": "Please complete all required steps before generating"}), 400

        # Generate prompt from selections
        prompt = generate_prompt_from_selections(selections)

        # Get dimensions from size and shape selections
        size_selection = selections.get('step3', '')
        shape_selection = selections.get('step4', '')
        width, height = get_dimensions_from_selection(size_selection, shape_selection)

        # Call Flux model synchronously and wait for result
        try:
            model = FluxModel()
            result = model.generate_image.remote(prompt, width, height)

            if result.get('error'):
                return jsonify({"error": result['error']}), 500

            # Store result for later download
            generated_images[session_id] = result

            return jsonify({
                "success": True,
                "session_id": session_id,
                "image_data": result.get('image_data'),
                "prompt": prompt
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route('/api/check-generation/<session_id>')
    def check_generation(session_id):
        if session_id in generated_images:
            return jsonify(generated_images[session_id])
        return jsonify({"status": "processing"})

    @flask_app.route('/api/download-image/<session_id>')
    def download_image(session_id):
        if session_id in generated_images and "image_data" in generated_images[session_id]:
            img_data = base64.b64decode(generated_images[session_id]["image_data"])
            from flask import send_file
            return send_file(
                io.BytesIO(img_data),
                mimetype='image/png',
                as_attachment=True,
                download_name=f'rug_design_{session_id}.png'
            )
        return jsonify({"error": "Image not found"}), 404

    @flask_app.route('/api/download-summary/<session_id>')
    def download_summary(session_id):
        if session_id not in user_sessions:
            return jsonify({"error": "Session not found"}), 404

        selections = user_sessions[session_id]
        summary = "CUSTOM RUG DESIGN SUMMARY\n"
        summary += "===========================\n\n"

        for i in range(1, len(STEP_DATA)):
            selection_id = selections.get(f'step{i}')
            if selection_id:
                step = STEP_DATA[i - 1]
                if step.get('type') == 'text_input':
                    summary += f"STEP {i}: {step['title']}\n"
                    summary += f"Details: {selection_id}\n\n"
                else:
                    option = next((opt for opt in step['options'] if opt['id'] == selection_id), None)
                    if option:
                        summary += f"STEP {i}: {step['title']}\n"
                        summary += f"Selected: {option['title']}\n"
                        summary += f"Description: {option['desc']}\n\n"

        if session_id in generated_images and "prompt" in generated_images[session_id]:
            summary += "\nGENERATION PROMPT:\n"
            summary += generated_images[session_id]["prompt"] + "\n"

        summary += "\nThank you for using our Custom Rug Design Tool!\n"

        return jsonify({"summary": summary})

    @flask_app.route('/api/reset-session/<session_id>', methods=['POST'])
    def reset_session(session_id):
        if session_id in user_sessions:
            user_sessions[session_id] = {}
        else:
            user_sessions[session_id] = {}

        if session_id in generated_images:
            del generated_images[session_id]

        return jsonify({"success": True})

    return flask_app