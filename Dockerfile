# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.5.1-base

# install custom nodes into comfyui (first node with --mode remote to fetch updated cache)
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

# unknown-registry custom nodes: clone known GitHub repos directly into /comfyui/custom_nodes
RUN git clone https://github.com/Aruntd008/comfyui_document_scanner /comfyui/custom_nodes/comfyui_document_scanner
RUN git clone https://github.com/Aruntd008/ComfyUI_SeamlessPattern /comfyui/custom_nodes/ComfyUI_SeamlessPattern
RUN git clone https://github.com/Aruntd008/ComfyUI_blender_render /comfyui/custom_nodes/ComfyUI_blender_render
# Note: many unknown_registry nodes (Reroute, Note, Label (rgthree), PrimitiveNode, Fast Groups Bypasser (rgthree), etc.) had no aux_id and could not be resolved automatically; skipped.

# download models into comfyui
RUN comfy model download --url https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors --relative-path models/style_models --filename flux1-redux-dev.safetensors
RUN comfy model download --url https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth --relative-path models/upscale_models --filename 4x-UltraSharp.pth
RUN comfy model download --url https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_K_S.gguf --relative-path models/diffusion_models --filename flux1-schnell-Q4_K_S.gguf
RUN comfy model download --url https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors --relative-path models/vae --filename ae.safetensors
RUN comfy model download --url https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors --relative-path models/text_encoders --filename t5xxl_fp8_e4m3fn.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/clip_l.safetensors --relative-path models/text_encoders --filename clip_l.safetensors
# The HuggingFace repo for the CLIP-Vision model stores the file as `model.safetensors`; download & rename to expected filename
RUN comfy model download --url https://huggingface.co/google/siglip2-so400m-patch16-512/resolve/main/model.safetensors --relative-path models/clip_vision --filename siglip2-so400m-patch16-512.safetensors

# copy all input data (like images or videos) into comfyui (uncomment and adjust if needed)
# COPY input/ /comfyui/input/
