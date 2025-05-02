"""
A small testing script to become familiar with StableDiffusion via GPUs prior to scaling up.
"""

from diffusers import StableDiffusionPipeline
import torch

def main():
    # 1) Make sure torch sees your GPU
    assert torch.cuda.is_available(), "CUDA not available—check your driver/CUDA install"
    for i in range(-10,21, 2):
        # Define parameters
        h = w = 512
        steps = 100
        guidance = i

        # 2) Load the SDv1.5 pipeline in half-precision
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        ).to("cuda")

        # 3) (Optional) Speed-ups for large models
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()  # offloads parts to CPU if you run out of VRAM

        # 4) Run a single prompt
        prompt = "A beautiful painting similar to Wanderer above the Sea of Fog"
        image = pipe(
            prompt,
            height=h,
            width=w,
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images[0]

        # 5) Save
        output_path = f"gpu_test_painting_{h}_{steps}_{guidance}.png"
        image.save(output_path)
        print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    main()
