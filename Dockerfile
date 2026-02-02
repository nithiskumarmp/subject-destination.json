# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.5.1-base

# =======================================================
# 1. INSTALL BLENDER, WGET & SYSTEM DEPENDENCIES
# =======================================================
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    blender \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# =======================================================
# 2. INSTALL PYTHON DEPENDENCIES
# =======================================================
RUN pip install --no-cache-dir numpy pillow opencv-python-headless

# =======================================================
# 3. INSTALL STANDARD CUSTOM NODES
# =======================================================
RUN comfy node install --exit-on-fail comfyui_essentials@1.1.0 --mode remote
RUN comfy node install --exit-on-fail ComfyUI_Comfyroll_CustomNodes
RUN comfy node install --exit-on-fail comfyui-kjnodes@1.2.9
RUN comfy node install --exit-on-fail was-node-suite-comfyui@1.0.2
RUN comfy node install --exit-on-fail comfyui-easy-use@1.3.6
RUN comfy node install --exit-on-fail ComfyUI-TiledDiffusion
RUN comfy node install --exit-on-fail comfyui-inpaint-cropandstitch@3.0.2
RUN comfy node install --exit-on-fail rgthree-comfy@1.0.2512112053
RUN comfy node install --exit-on-fail comfyui-rmbg@3.0.0
RUN comfy node install --exit-on-fail comfyui_layerstyle@2.0.38
RUN comfy node install --exit-on-fail ComfyUI_AdvancedRefluxControl

# =======================================================
# 4. COPY LOCAL CUSTOM NODES (From your GitHub Repo)
# =======================================================
COPY comfyui_document_scanner /comfyui/custom_nodes/comfyui_document_scanner
COPY ComfyUI_SeamlessPattern /comfyui/custom_nodes/ComfyUI_SeamlessPattern
COPY ComfyUI_blender_render /comfyui/custom_nodes/ComfyUI_blender_render

# =======================================================
# 5. DOWNLOAD MODELS (Using wget - Matches your list)
# =======================================================

# 1. FLUX T5XXL Text Encoder
RUN wget -O /comfyui/models/clip/t5xxl_fp16.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors

# 2. FLUX CLIP-L Text Encoder
RUN wget -O /comfyui/models/clip/clip_l.safetensors https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors

# 3. FLUX VAE Model
RUN wget -O /comfyui/models/vae/ae.safetensors https://huggingface.co/camenduru/FLUX.1-dev/resolve/d616d290809ffe206732ac4665a9ddcdfb839743/ae.safetensors

# 4. SigLIP2 Vision Model
RUN wget -O /comfyui/models/clip_vision/sglip2-so400m-patch16-512.safetensors https://huggingface.co/google/siglip2-so400m-patch16-512/resolve/main/model.safetensors

# 5. FLUX Redux Style Model
RUN wget -O /comfyui/models/style_models/flux1-redux-dev.safetensors https://huggingface.co/camenduru/FLUX.1-dev/resolve/d616d290809ffe206732ac4665a9ddcdfb839743/flux1-redux-dev.safetensors

# 6. FLUX UNet Diffusion Model
# ⚠️ RENAMING: Saving 'unet_fp8.safetensors' as 'flux1-dev.safetensors'
# This ensures Node 282 in your JSON finds the file it expects.
RUN wget -O /comfyui/models/diffusion_models/flux1-dev.safetensors https://huggingface.co/yichengup/flux.1-fill-dev-OneReward/resolve/main/unet_fp8.safetensors

# 7. UltraSharp (Upscaler)
RUN wget -O /comfyui/models/upscale_models/4x-UltraSharp.pth https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth

# 8. Extra Flux FP8 (Safety for Node 292 if workflow logic branches)
RUN wget -O /comfyui/models/diffusion_models/flux1-dev-fp8-e4m3fn.safetensors https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8-e4m3fn.safetensors

# =======================================================
# 6. DOWNLOAD BLEND FILE & IMAGES
# =======================================================
RUN wget -O /comfyui/input/file.blend https://huggingface.co/Srivarshan7/my-assets/resolve/b61a31e/file.blend

COPY scene_destination.png /comfyui/input/
COPY curtain_mask.png /comfyui/input/
COPY processed_IMG20250919150037.jpg /comfyui/input/
