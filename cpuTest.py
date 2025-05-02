"""
A small testing script to become familiar with StableDiffusion via CPUs prior to scaling up.
"""

from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")
pipe.enable_attention_slicing()

if torch.cuda.is_available():
    pipe.enable_model_cpu_offload()

# generate a test
image = pipe(
    "A beautiful painting similar to Wanderer above the Sea of Fog",
    height=512, width=512,
    num_inference_steps=50,
    guidance_scale=8
).images[0]

image.save("cpu_test.png")
