"""
This script uses Stable Diffusion XL (SDXL) to generate and iteratively mutate images based on votes.
It is optimized to run on a HPC cluster.
"""

import os
import sys
from collections import Counter

import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)

class ImageGenerationPipeline:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vote_threshold: int = 3
    ):
        # We assume at least one CUDA GPU is available, and with two A100s we'll use device_map="auto"
        assert torch.cuda.is_available(), "CUDA not available—please run on GPU nodes."
        
        # Load SDXL text-to-image, split across GPUs
        self.sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,    # full precision
            device_map="auto"             # auto-shard across all available GPUs
        )
        self.sdxl_pipe.enable_xformers_memory_efficient_attention()

        # Load SDXL image-to-image, likewise sharded
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        self.img2img_pipe.enable_xformers_memory_efficient_attention()

        self.vote_threshold = vote_threshold
        self.image_sets = []

    def generate_initial_sets(
        self,
        prompts: list[str],
        num_variations: int = 3,
        output_dir: str = "outputs",
        guidance_scale: float = 9.0,
        num_inference_steps: int = 50
    ) -> list[list[str]]:
        os.makedirs(output_dir, exist_ok=True)
        self.image_sets = []

        for idx, prompt in enumerate(prompts):
            set_dir = os.path.join(output_dir, f"set_{idx+1}")
            os.makedirs(set_dir, exist_ok=True)
            paths = []

            for j in range(num_variations):
                out = self.sdxl_pipe(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
                img = out.images[0]
                fn = f"img_{j+1}.png"
                path = os.path.join(set_dir, fn)
                img.save(path)
                paths.append(path)

            self.image_sets.append(paths)

        self.save_gallery(output_dir)
        return self.image_sets

    def save_gallery(
        self,
        output_dir: str,
        gallery_name: str = "gallery.png"
    ):
        rows = []
        for paths in self.image_sets:
            imgs = [Image.open(p).convert("RGB") for p in paths]
            widths, heights = zip(*(im.size for im in imgs))
            total_width = sum(widths)
            max_height = max(heights)
            row = Image.new("RGB", (total_width, max_height))
            x = 0
            for im in imgs:
                row.paste(im, (x, 0))
                x += im.width
            rows.append(row)

        if not rows:
            return

        w = rows[0].width
        h = sum(r.height for r in rows)
        gallery = Image.new("RGB", (w, h))
        y = 0
        for r in rows:
            gallery.paste(r, (0, y))
            y += r.height

        path = os.path.join(output_dir, gallery_name)
        gallery.save(path)
        print(f"✧ Gallery saved to {path}")

    def display_and_vote(self) -> int | None:
        print("Vote: keys 1–9 (q to quit).")
        votes = Counter()
        total = len(self.image_sets) * len(self.image_sets[0])

        while True:
            k = input("Your vote: ").strip().lower()
            if k == "q":
                return None
            if k.isdigit() and 1 <= (i := int(k)) <= total:
                votes[i] += 1
                print(f"  ✓ image {i} → {votes[i]} votes")
                if votes[i] >= self.vote_threshold:
                    return i
            else:
                print(f"  ✗ Enter 1–{total} or q")

    def iterate(
        self,
        selected_key: int,
        prompts_variation: list[str],
        num_variations: int = 3,
        output_dir: str = "outputs",
        strength: float = 0.85,
        guidance_scale: float = 10.0,
        num_inference_steps: int = 70
    ) -> list[str]:
        set_i = (selected_key - 1) // num_variations
        img_i = (selected_key - 1) % num_variations
        src = self.image_sets[set_i][img_i]

        init_img = Image.open(src).convert("RGB")
        out_dir = os.path.join(output_dir, f"set_{set_i+1}_iter")
        os.makedirs(out_dir, exist_ok=True)

        new_paths = []
        for j, prompt in enumerate(prompts_variation):
            out = self.img2img_pipe(
                prompt=prompt,
                init_image=init_img,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            img = out.images[0]
            fn = f"img_{j+1}.png"
            path = os.path.join(out_dir, fn)
            img.save(path)
            new_paths.append(path)

        self.image_sets[set_i] = new_paths
        self.save_gallery(output_dir)
        return new_paths


if __name__ == "__main__":
    base_prompts = [
        "A tropical beach at sunset",
        "A serene mountain landscape",
        "A futuristic city at night"
    ]
    pipeline = ImageGenerationPipeline(vote_threshold=3)
    pipeline.generate_initial_sets(base_prompts)

    while True:
        sel = pipeline.display_and_vote()
        if sel is None:
            print("Goodbye!")
            sys.exit()

        parent_prompt = base_prompts[(sel-1)//3]
        variations = [
            f"{parent_prompt}"
            for _ in range(3)
        ]
        pipeline.iterate(sel, variations)
