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
    def generate_image(self, prompt: str, width: int = 896, height: int = 1200, seed: int = None, enable_seamless: bool = True):
        """Generate an image from a prompt with H100 optimizations and seamless pattern techniques"""
        import torch
        from io import BytesIO
        import gc
        import random

        try:
            # Generate random seed if not provided
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            print(f"Generating image with prompt:\n{prompt}")
            print(f"Negative prompt:\n{DEFAULT_NEGATIVE_PROMPT}")
            print(f"Image dimensions: {width}x{height}")
            print(f"Seed: {seed}")
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
                        max_sequence_length=512,
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
                        max_sequence_length=512,
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

    # Determine base size multiplier from size selection
    if 'small' in size_selection_lower or '120' in size_selection:
        size_mult = 0.75  # smaller dimensions
    elif 'large' in size_selection_lower or '340' in size_selection or '290' in size_selection:
        size_mult = 1.25  # larger dimensions
    else:
        size_mult = 1.0  # medium/default

    # Shape-based dimensions (width, height)
    shape_dimensions = {
        'circle': (1024, 1024),      # 1:1 square for circular design
        'square': (1024, 1024),      # 1:1 square
        'oval': (1024, 768),         # 4:3 horizontal ellipse
        'rectangle': (896, 1200),    # ~3:4 vertical rectangle
        'runner': (1280, 512),       # 5:2 long narrow
        'round at ends': (1024, 1024), # stadium shape ~9:5
        'octagon': (1024, 1024),     # 1:1 square for octagon
    }

    # Find matching shape
    base_width, base_height = (896, 1152)  # default rectangle
    for shape_key, dims in shape_dimensions.items():
        if shape_key in shape_selection_lower:
            base_width, base_height = dims
            break

    # Apply size multiplier and ensure divisible by 16
    width = int((base_width * size_mult) // 16) * 16
    height = int((base_height * size_mult) // 16) * 16

    # Clamp to reasonable bounds (min 512, max 1536)
    width = max(512, min(1536, width))
    height = max(512, min(1536, height))

    return (width, height)


def generate_prompt_from_selections(selections, session_id=None, style_configs=None):
    """Convert user selections into a detailed prompt for FLUX.1-schnell

    Step order matches rug-options.json accordions:
    - step1 = shape
    - step2 = size
    - step3 = color
    - step4 = design-style
    - step5 = design-details
    - step6 = rooms
    """

    # Step 1: Shape
    shape = selections.get('step1', 'Rectangle').lower()

    # Step 2: Size (used for image dimensions, not in prompt)

    # Step 3: Color
    color = selections.get('step3', '')

    # Step 4: Design Style
    style = selections.get('step4', 'custom')

    # Step 5: Design Details
    detail = selections.get('step5', '')

    # Step 6: Room/Location
    room = selections.get('step6', 'indoor space')

    # Build shape description for rug
    shape_descriptions = {
        'rectangle': 'rectangular',
        'square': 'square',
        'circle': 'circular',
        'oval': 'oval',
        'runner': 'runner',
        'round at ends': 'narrow and round at ends',
        'octagon': 'octagonal'
    }
    shape_name = shape_descriptions.get(shape, shape)

    # Determine which prompt structure to use based on shape
    if shape in ['circle', 'oval', 'octagon', 'round at ends']:
        # ORIGINAL PROMPT STRUCTURE for circle and oval
        prompt = f"Flat 2D vector illustration of a {shape_name} rug. "

        # Handle Abstract style with randomization
        if style == 'Abstract' and session_id and style_configs:
            abstract_config = style_configs.get('Abstract', {})
            if abstract_config:
                # Get randomized parameters
                params = select_random_abstract_params(session_id, abstract_config)

                if params:
                    # Build enhanced prompt with abstract parameters
                    style_description = (
                        f"{params['fluidStyles']} in {params['patterns']} pattern "
                        f"with {params['composition']} arrangement, "
                        f"{params['density']} density, {params['scale']}, "
                        f"{params['orientation']}, {params['spacing']}, "
                        f"{params['arrangement']}"
                    )
                    prompt += f"The design displays {style_description} design"
                else:
                    prompt += f"The design displays {style} design"
            else:
                prompt += f"The design displays {style} design"
        else:
            prompt += f"The design displays {style} design"

        # Add detail if provided
        if detail:
            prompt += f", {detail}"

        # Add color if provided
        if color:
            prompt += f", {color} colors"

        prompt += ".\n\n"

        # Add shape clarification
        prompt += f"The shape of the design should be like {shape_name} rug.\n"

        # Add shadow negatives
        prompt += """No cast shadow, no drop shadow, no ambient shadow, no halo, no outline shadow, no separation from background. no shading, no lighting, no realism, no depth, no shadows, no 3D effects.

Strict vector artwork:
solid flat color fills only,
hard edges, sharp geometry,

NO texture of any kind:
no fabric texture, no grain, no noise,

SVG / AI / EPS style,
screen-print ready, textile CAD pattern,
manufacturing-ready vector artwork.
"""
    else:
        # NEW PROMPT STRUCTURE for rectangle, square, runner, round at ends, octagon
        if style == 'Abstract' and session_id and style_configs:
            abstract_config = style_configs.get('Abstract', {})
            if abstract_config:
                # Get randomized parameters
                params = select_random_abstract_params(session_id, abstract_config)

                if params:
                    # Build enhanced prompt with abstract parameters
                    style_description = (
                        f"{params['fluidStyles']} in {params['patterns']} pattern "
                        f"with {params['composition']} arrangement, "
                        f"{params['density']} density, {params['scale']}, "
                        f"{params['orientation']}, {params['spacing']}, "
                        f"{params['arrangement']}"
                    )

                    prompt = f"Flat 2D vector illustration of {style_description} "
                    prompt += f"rug design suitable for a {room}"
                else:
                    # Fallback if parameters unavailable
                    prompt = f"Flat 2D vector illustration of {style} rug design suitable for a {room}"
            else:
                # Fallback if config not found
                prompt = f"Flat 2D vector illustration of {style} rug design suitable for a {room}"
        else:
            # Original logic for non-Abstract styles
            prompt = f"Flat 2D vector illustration of {style} rug design suitable for a {room}"

        if shape:
            prompt += f" in {shape_name} shape"

        if detail:
            prompt += f", {detail}"

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


def select_random_abstract_params(session_id, style_configs):
    """
    Select random abstract parameters that haven't been used yet.

    Args:
        session_id: User session identifier
        style_configs: Abstract parameter configurations from JSON

    Returns:
        dict: {fluidStyles, composition, patterns, density, scale, orientation, spacing, arrangement} or None if exhausted
    """
    import random

    # Initialize tracking if not exists
    if session_id not in user_sessions:
        user_sessions[session_id] = {}

    if 'abstract_params_tracking' not in user_sessions[session_id]:
        # Calculate total combinations based on available parameters
        total_combos = (
            len(style_configs.get('fluidStyles', [])) *
            len(style_configs.get('composition', [])) *
            len(style_configs.get('patterns', []))
        )

        user_sessions[session_id]['abstract_params_tracking'] = {
            'used_combinations': set(),  # Use set for O(1) lookup
            'current_params': None,
            'total_combinations': total_combos,
            'used_count': 0
        }

    tracking = user_sessions[session_id]['abstract_params_tracking']
    used_combinations = tracking['used_combinations']

    # Check if all combinations exhausted - auto-reset
    if tracking['used_count'] >= tracking['total_combinations']:
        tracking['used_combinations'] = set()
        tracking['used_count'] = 0
        tracking['reset_count'] = tracking.get('reset_count', 0) + 1
        used_combinations = tracking['used_combinations']

    # Generate available combinations
    max_attempts = 100  # Prevent infinite loop
    attempts = 0

    while attempts < max_attempts:
        # Select one parameter from each category
        fluidStyle = random.choice(style_configs.get('fluidStyles', ['liquid paint pour']))
        comp = random.choice(style_configs.get('composition', ['centrally composed']))
        pattern = random.choice(style_configs.get('patterns', ['liquid marble effect']))
        density = random.choice(style_configs.get('density', ['moderately filled']))
        scale = random.choice(style_configs.get('scale', ['uniform scale']))
        orientation = random.choice(style_configs.get('orientation', ['horizontal direction']))
        spacing = random.choice(style_configs.get('spacing', ['tight uniform gaps']))
        arrangement = random.choice(style_configs.get('arrangement', ['organic natural flow']))

        # Create unique key for this combination (using main parameters)
        combo_key = f"{fluidStyle}|{comp}|{pattern}"

        if combo_key not in used_combinations:
            # Track usage
            used_combinations.add(combo_key)
            selected = {
                'fluidStyles': fluidStyle,
                'composition': comp,
                'patterns': pattern,
                'density': density,
                'scale': scale,
                'orientation': orientation,
                'spacing': spacing,
                'arrangement': arrangement
            }
            tracking['current_params'] = selected
            tracking['used_count'] = len(used_combinations)

            return selected

        attempts += 1

    # Fallback: if max attempts reached, reset and try once more
    tracking['used_combinations'] = set()
    tracking['used_count'] = 0
    return {
        'fluidStyles': random.choice(style_configs.get('fluidStyles', ['liquid paint pour'])),
        'composition': random.choice(style_configs.get('composition', ['centrally composed'])),
        'patterns': random.choice(style_configs.get('patterns', ['liquid marble effect'])),
        'density': random.choice(style_configs.get('density', ['moderately filled'])),
        'scale': random.choice(style_configs.get('scale', ['uniform scale'])),
        'orientation': random.choice(style_configs.get('orientation', ['horizontal direction'])),
        'spacing': random.choice(style_configs.get('spacing', ['tight uniform gaps'])),
        'arrangement': random.choice(style_configs.get('arrangement', ['organic natural flow']))
    }


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
            min-height: 0;
            position: relative;
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
            min-height: 0;
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
            overflow: hidden;
        }

        .image-display.show {
            display: flex;
            flex-direction: column;
        }

        .image-display:fullscreen {
            background: #1f2937;
        }

        .image-display:fullscreen .image-wrapper {
            background: #1f2937;
        }

        .image-display:-webkit-full-screen {
            background: #1f2937;
        }

        .image-display:-webkit-full-screen .image-wrapper {
            background: #1f2937;
        }

        .image-wrapper {
            width: 100%;
            height: 100%;
            overflow: auto;
            display: flex;
            align-items: flex-start;
            justify-content: flex-start;
            position: relative;
        }

        .image-wrapper.centered {
            align-items: center;
            justify-content: center;
        }

        .image-wrapper.grabbing {
            cursor: grabbing;
        }

        .generated-image {
            transition: width 0.2s ease, height 0.2s ease;
            display: block;
            max-width: none;
            max-height: none;
            margin: auto;
        }

        .image-actions {
            position: absolute;
            top: 16px;
            right: 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            z-index: 10;
            pointer-events: auto;
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

        #fullscreen-btn {
            margin-bottom: 8px;
            border-bottom: 2px solid var(--border);
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
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }

        .choice-item:last-child {
            border-bottom: none;
        }

        .choice-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            color: var(--dark);
            font-size: 13px;
            flex-shrink: 0;
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
            color: var(--dark);
            font-size: 13px;
            line-height: 1.4;
            text-align: right;
            flex: 1;
        }

        .empty-choice {
            color: var(--gray);
            font-style: italic;
            font-size: 12px;
        }

        .prompt-section {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 2px solid var(--border);
        }

        .prompt-header {
            font-size: 13px;
            font-weight: 600;
            color: var(--dark);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .prompt-textarea {
            width: 100%;
            min-height: 200px;
            padding: 12px;
            border: 1.5px solid var(--border);
            border-radius: 8px;
            font-size: 12px;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            resize: vertical;
            transition: var(--transition);
            background: white;
            color: var(--dark);
            line-height: 1.5;
        }

        .prompt-textarea:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.08);
        }

        .prompt-textarea:disabled {
            background: var(--gray-light);
            cursor: not-allowed;
        }

        .confirm-btn {
            width: 100%;
            margin-top: 10px;
            padding: 12px;
            background: var(--secondary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .confirm-btn:hover:not(:disabled) {
            background: #0f766e;
            transform: translateY(-1px);
        }

        .confirm-btn:disabled {
            background: var(--gray);
            cursor: not-allowed;
            transform: none;
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
                    <div class="image-wrapper" id="image-wrapper">
                        <img class="generated-image" id="generated-image" src="" alt="Generated Rug Design">
                    </div>
                    <div class="image-actions">
                        <button class="image-action-btn" id="zoom-in-btn" title="Zoom In">
                            <i class="fas fa-search-plus"></i>
                        </button>
                        <button class="image-action-btn" id="zoom-out-btn" title="Zoom Out">
                            <i class="fas fa-search-minus"></i>
                        </button>
                        <button class="image-action-btn" id="zoom-reset-btn" title="Reset Zoom">
                            <i class="fas fa-expand"></i>
                        </button>
                        <button class="image-action-btn" id="fullscreen-btn" title="Fullscreen">
                            <i class="fas fa-expand-arrows-alt"></i>
                        </button>
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

                <div class="prompt-section" id="prompt-section" style="display: none;">
                    <div class="prompt-header">
                        <i class="fas fa-wand-magic-sparkles"></i>
                        <span>Generated Prompt</span>
                    </div>
                    <textarea class="prompt-textarea" id="prompt-textarea" placeholder="Click 'Generate Rug Design' to create your prompt..."></textarea>
                    <button class="confirm-btn" id="confirm-generate-btn" disabled>
                        <i class="fas fa-check-circle"></i>
                        Confirm & Generate Image
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let stepData = [];
        let userChoices = {};
        let sessionId = '';
        let currentStep = 0;
        let zoomLevel = 1;
        let fullscreenZoomLevel = 1;
        let fitZoomLevel = 1;
        let fullscreenFitZoomLevel = 1;
        let isFullscreen = false;
        let isPanning = false;
        let startX, startY, scrollLeft, scrollTop;

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

            document.getElementById('generate-btn').addEventListener('click', generatePrompt);
            document.getElementById('prev-btn').addEventListener('click', () => navigateToStep(currentStep - 1));
            document.getElementById('next-btn').addEventListener('click', () => navigateToStep(currentStep + 1));
            document.getElementById('confirm-generate-btn').addEventListener('click', confirmAndGenerate);
            document.getElementById('download-btn').addEventListener('click', downloadGeneratedImage);
            document.getElementById('refresh-btn').addEventListener('click', generatePrompt);

            // Zoom controls
            document.getElementById('zoom-in-btn').addEventListener('click', zoomIn);
            document.getElementById('zoom-out-btn').addEventListener('click', zoomOut);
            document.getElementById('zoom-reset-btn').addEventListener('click', resetZoom);
            document.getElementById('fullscreen-btn').addEventListener('click', toggleFullscreen);

            // Mouse wheel zoom
            const imageWrapper = document.getElementById('image-wrapper');
            imageWrapper.addEventListener('wheel', handleMouseWheel, { passive: false });

            // Pan functionality
            imageWrapper.addEventListener('mousedown', startPan);
            imageWrapper.addEventListener('mousemove', pan);
            imageWrapper.addEventListener('mouseup', endPan);
            imageWrapper.addEventListener('mouseleave', endPan);
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
                        <span>${accordion.title}:</span>
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

            // Only require step4 (Design Style) to be selected
            const designStyleSelected = stepData.length > 3 && userChoices[stepData[3].id];
            generateBtn.disabled = !designStyleSelected;
        }

        async function generatePrompt() {
            const generateBtn = document.getElementById('generate-btn');
            const promptSection = document.getElementById('prompt-section');
            const promptTextarea = document.getElementById('prompt-textarea');
            const confirmBtn = document.getElementById('confirm-generate-btn');

            const originalText = generateBtn.innerHTML;
            generateBtn.innerHTML = '<div class="loading-spinner"></div> Generating Prompt...';
            generateBtn.disabled = true;

            try {
                const selections = {};
                stepData.forEach((step, index) => {
                    selections[`step${index + 1}`] = userChoices[step.id];
                });

                const response = await fetch('/api/generate-prompt', {
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
                } else if (data.prompt) {
                    promptTextarea.value = data.prompt;
                    promptSection.style.display = 'block';
                    confirmBtn.disabled = false;
                }

            } catch (error) {
                console.error('Error:', error);
                alert('Failed to generate prompt.');
            }

            generateBtn.innerHTML = originalText;
            generateBtn.disabled = false;
        }

        async function confirmAndGenerate() {
            const confirmBtn = document.getElementById('confirm-generate-btn');
            const promptTextarea = document.getElementById('prompt-textarea');
            const imagePlaceholder = document.getElementById('image-placeholder');
            const imageDisplay = document.getElementById('image-display');
            const imageStatus = document.getElementById('image-status');
            const generatedImage = document.getElementById('generated-image');

            const customPrompt = promptTextarea.value.trim();
            if (!customPrompt) {
                alert('Please enter a prompt before generating.');
                return;
            }

            const originalText = confirmBtn.innerHTML;
            confirmBtn.innerHTML = '<div class="loading-spinner"></div> Generating Image...';
            confirmBtn.disabled = true;
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
                        selections: selections,
                        custom_prompt: customPrompt
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

                    // Calculate and apply fit zoom when image is loaded
                    generatedImage.onload = function() {
                        fitZoomLevel = calculateFitZoom();
                        zoomLevel = fitZoomLevel;
                        applyZoom();
                    };
                }

            } catch (error) {
                console.error('Error:', error);
                alert('Failed to generate design.');
                imageStatus.textContent = 'Generation failed';
            }

            confirmBtn.innerHTML = originalText;
            confirmBtn.disabled = false;
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

        // Zoom functions
        function zoomIn() {
            if (isFullscreen) {
                fullscreenZoomLevel = Math.min(fullscreenZoomLevel + 0.25, 5);
            } else {
                zoomLevel = Math.min(zoomLevel + 0.25, 5);
            }
            applyZoom();
        }

        function zoomOut() {
            if (isFullscreen) {
                fullscreenZoomLevel = Math.max(fullscreenZoomLevel - 0.25, 0.5);
            } else {
                zoomLevel = Math.max(zoomLevel - 0.25, 0.5);
            }
            applyZoom();
        }

        function resetZoom() {
            if (isFullscreen) {
                fullscreenZoomLevel = fullscreenFitZoomLevel;
            } else {
                zoomLevel = fitZoomLevel;
            }
            applyZoom();
        }

        function calculateFitZoom() {
            const imageWrapper = document.getElementById('image-wrapper');
            const generatedImage = document.getElementById('generated-image');

            if (!generatedImage.naturalWidth || !generatedImage.naturalHeight) {
                return 1;
            }

            const wrapperWidth = imageWrapper.clientWidth;
            const wrapperHeight = imageWrapper.clientHeight;
            const imageWidth = generatedImage.naturalWidth;
            const imageHeight = generatedImage.naturalHeight;

            // Account for margin (20px on each side)
            const margin = 40;

            // Calculate scale to fit both width and height
            const scaleX = (wrapperWidth - margin) / imageWidth;
            const scaleY = (wrapperHeight - margin) / imageHeight;

            // Use the smaller scale to ensure the entire image fits
            // Allow scaling up to 1.5x if image is small, but cap at that
            return Math.min(scaleX, scaleY, 1.5);
        }

        function applyZoom() {
            const image = document.getElementById('generated-image');
            const imageWrapper = document.getElementById('image-wrapper');

            // Use the correct zoom level based on current mode
            const currentZoom = isFullscreen ? fullscreenZoomLevel : zoomLevel;

            // Calculate actual dimensions
            const width = image.naturalWidth * currentZoom;
            const height = image.naturalHeight * currentZoom;

            // Apply dimensions
            image.style.width = width + 'px';
            image.style.height = height + 'px';

            // Center the image if it fits within the wrapper, otherwise align to top-left
            const wrapperWidth = imageWrapper.clientWidth;
            const wrapperHeight = imageWrapper.clientHeight;

            if (width <= wrapperWidth && height <= wrapperHeight) {
                imageWrapper.classList.add('centered');
                image.style.margin = 'auto';
                imageWrapper.style.cursor = 'default';
            } else {
                imageWrapper.classList.remove('centered');
                image.style.margin = '0';
                imageWrapper.style.cursor = 'grab';
            }
        }

        function handleMouseWheel(e) {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            if (isFullscreen) {
                fullscreenZoomLevel = Math.max(0.5, Math.min(5, fullscreenZoomLevel + delta));
            } else {
                zoomLevel = Math.max(0.5, Math.min(5, zoomLevel + delta));
            }
            applyZoom();
        }

        // Pan functions
        function startPan(e) {
            const wrapper = document.getElementById('image-wrapper');
            const image = document.getElementById('generated-image');

            // Only enable panning if the image is larger than the wrapper
            const canPan = image.scrollWidth > wrapper.clientWidth || image.scrollHeight > wrapper.clientHeight;

            if (canPan) {
                isPanning = true;
                wrapper.classList.add('grabbing');
                startX = e.clientX;
                startY = e.clientY;
                scrollLeft = wrapper.scrollLeft;
                scrollTop = wrapper.scrollTop;
                e.preventDefault();
            }
        }

        function pan(e) {
            if (!isPanning) return;
            e.preventDefault();

            const wrapper = document.getElementById('image-wrapper');
            const walkX = e.clientX - startX;
            const walkY = e.clientY - startY;

            wrapper.scrollLeft = scrollLeft - walkX;
            wrapper.scrollTop = scrollTop - walkY;
        }

        function endPan() {
            const wrapper = document.getElementById('image-wrapper');
            isPanning = false;
            wrapper.classList.remove('grabbing');
        }

        // Fullscreen function
        function toggleFullscreen() {
            const imageContainer = document.getElementById('image-display');

            if (!document.fullscreenElement) {
                if (imageContainer.requestFullscreen) {
                    imageContainer.requestFullscreen();
                } else if (imageContainer.webkitRequestFullscreen) {
                    imageContainer.webkitRequestFullscreen();
                } else if (imageContainer.msRequestFullscreen) {
                    imageContainer.msRequestFullscreen();
                }
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                } else if (document.msExitFullscreen) {
                    document.msExitFullscreen();
                }
            }
        }

        // Update fullscreen button icon when entering/exiting fullscreen
        document.addEventListener('fullscreenchange', updateFullscreenButton);
        document.addEventListener('webkitfullscreenchange', updateFullscreenButton);
        document.addEventListener('msfullscreenchange', updateFullscreenButton);

        function updateFullscreenButton() {
            const fullscreenBtn = document.getElementById('fullscreen-btn');
            const icon = fullscreenBtn.querySelector('i');

            if (document.fullscreenElement) {
                isFullscreen = true;
                icon.className = 'fas fa-compress-arrows-alt';
                fullscreenBtn.title = 'Exit Fullscreen';

                // Calculate fit zoom for fullscreen mode
                fullscreenFitZoomLevel = calculateFitZoom();
                // Set fullscreen zoom to fit if it's the first time entering fullscreen
                if (fullscreenZoomLevel === 1) {
                    fullscreenZoomLevel = fullscreenFitZoomLevel;
                }
            } else {
                isFullscreen = false;
                icon.className = 'fas fa-expand-arrows-alt';
                fullscreenBtn.title = 'Fullscreen';

                // Recalculate fit zoom for normal mode
                fitZoomLevel = calculateFitZoom();
            }

            // Apply the zoom for the current mode
            applyZoom();
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

    @flask_app.route('/api/generate-prompt', methods=['POST'])
    def generate_prompt_endpoint():
        import json

        data = request.json
        session_id = data.get('session_id', str(uuid.uuid4()))
        selections = data.get('selections')

        if not selections:
            if session_id not in user_sessions:
                return jsonify({"error": "No selections found for this session"}), 400
            selections = user_sessions[session_id]
        else:
            user_sessions[session_id] = selections

        # Only require step4 (Design Style) to be selected
        if not selections.get('step4'):
            return jsonify({"error": "Please select a Design Style before generating"}), 400

        # Load style configurations from JSON
        style_configs = None
        try:
            with open('/app/rug-options.json', 'r') as f:
                data_json = json.load(f)
                style_configs = data_json.get('styleConfigurations', {})
        except Exception as e:
            print(f"Warning: Could not load style configurations: {e}")

        # Generate prompt with style configurations
        prompt = generate_prompt_from_selections(selections, session_id, style_configs)

        # Get tracking info for Abstract style
        tracking_info = {}
        if selections.get('step4') == 'Abstract' and session_id in user_sessions:
            tracking = user_sessions[session_id].get('abstract_params_tracking', {})
            if tracking:
                tracking_info = {
                    'used_count': tracking.get('used_count', 0),
                    'total_combinations': tracking.get('total_combinations', 0),
                    'current_params': tracking.get('current_params', {}),
                    'reset_count': tracking.get('reset_count', 0)
                }

        return jsonify({
            "success": True,
            "session_id": session_id,
            "prompt": prompt,
            "tracking_info": tracking_info
        })

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

        # Only require step4 (Design Style) to be selected
        if not selections.get('step4'):
            return jsonify({"error": "Please select a Design Style before generating"}), 400

        # Use custom prompt if provided, otherwise generate from selections
        custom_prompt = data.get('custom_prompt')
        if custom_prompt:
            prompt = custom_prompt
        else:
            # Load style configurations from JSON
            import json
            style_configs = None
            try:
                with open('/app/rug-options.json', 'r') as f:
                    data_json = json.load(f)
                    style_configs = data_json.get('styleConfigurations', {})
            except Exception as e:
                print(f"Warning: Could not load style configurations: {e}")

            prompt = generate_prompt_from_selections(selections, session_id, style_configs)

        # Get dimensions from size and shape selections
        size_selection = selections.get('step2', '')
        shape_selection = selections.get('step1', '')
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

        # Step labels for better readability
        step_labels = {
            'step1': 'Shape',
            'step2': 'Size',
            'step3': 'Color',
            'step4': 'Design Style',
            'step5': 'Design Details',
            'step6': 'Room/Location'
        }

        for step_key, label in step_labels.items():
            if step_key in selections and selections[step_key]:
                summary += f"{label}: {selections[step_key]}\n"

        if session_id in generated_images and "prompt" in generated_images[session_id]:
            summary += "\n" + "="*27 + "\n"
            summary += "GENERATION PROMPT:\n"
            summary += "="*27 + "\n"
            summary += generated_images[session_id]["prompt"] + "\n"

        summary += "\nThank you for using RugWise Studio!\n"

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