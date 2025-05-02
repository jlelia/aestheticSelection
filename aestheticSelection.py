"""
The purpose of this script is to generate increasingly pleasing images using Stable Diffusion to 
introduce slight variation (mutations) and voters to choose their favorite of the variants (selection).
"""

import os
import sys
import time
from collections import Counter
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

class ImageGenerationPipeline:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None, vote_threshold=3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.sd_pipe = self.sd_pipe.to(self.device)
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(self.device)
        self.vote_threshold = vote_threshold
        self.image_sets = []

    def generate_initial_sets(self, prompts, num_variations=3, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        self.image_sets = []
        for idx, prompt in enumerate(prompts):
            set_dir = os.path.join(output_dir, f"set_{idx+1}")
            os.makedirs(set_dir, exist_ok=True)
            images = []
            for j in range(num_variations):
                img = self.sd_pipe(prompt, guidance_scale=7.5).images[0]
                path = os.path.join(set_dir, f"img_{j+1}.png")
                img.save(path)
                images.append(path)
            self.image_sets.append(images)
        self.save_gallery(output_dir)
        return self.image_sets

    def save_gallery(self, output_dir, gallery_name="gallery.png"):
        # Create a grid of current image sets: rows = sets, cols = images per set
        rows = []
        for images in self.image_sets:
            imgs = [Image.open(p).convert("RGB") for p in images]
            widths, heights = zip(*(i.size for i in imgs))
            total_width = sum(widths)
            max_height = max(heights)
            row_img = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in imgs:
                row_img.paste(im, (x_offset, 0))
                x_offset += im.width
            rows.append(row_img)
        # combine rows
        gallery_width = rows[0].width if rows else 0
        gallery_height = sum(r.height for r in rows)
        gallery = Image.new('RGB', (gallery_width, gallery_height))
        y_offset = 0
        for r in rows:
            gallery.paste(r, (0, y_offset))
            y_offset += r.height
        gallery_path = os.path.join(output_dir, gallery_name)
        gallery.save(gallery_path)
        try:
            gallery.show()
        except Exception:
            pass
        print(f"Gallery saved to {gallery_path}")

    def display_and_vote(self):
        print("Please vote for your preferred image using keys 1-9:")
        print("Sets 1: 1-3, Set 2: 4-6, Set 3: 7-9")
        votes = Counter()
        while True:
            key = input("Your vote (q to quit): ").strip()
            if key.lower() == 'q':
                break
            if key.isdigit():
                k = int(key)
                if 1 <= k <= len(self.image_sets)*len(self.image_sets[0]):
                    votes[k] += 1
                    print(f"Vote recorded for image {k}. Total: {votes[k]}")
                    if votes[k] >= self.vote_threshold:
                        return k
                else:
                    print("Invalid key range.")
            else:
                print("Invalid input.")
        return None

    def iterate(self, selected_key, prompts_variation, num_variations=3, output_dir="outputs"):
        set_idx = (selected_key - 1) // num_variations
        img_idx = (selected_key - 1) % num_variations
        selected_img_path = self.image_sets[set_idx][img_idx]
        init_image = Image.open(selected_img_path).convert("RGB")
        next_set_dir = os.path.join(output_dir, f"set_{set_idx+1}_iter")
        os.makedirs(next_set_dir, exist_ok=True)
        new_paths = []
        for j, prompt in enumerate(prompts_variation):
            out = self.img2img_pipe(
                prompt=prompt,
                image=init_image,    
                strength=.85,
                guidance_scale=7
            )
            img = out.images[0]
            path = os.path.join(next_set_dir, f"img_{j+1}.png")
            img.save(path)
            new_paths.append(path)
        self.image_sets[set_idx] = new_paths
        self.save_gallery(output_dir)
        return new_paths

if __name__ == "__main__":
    prompts = ["A beautiful tropical beach", "A serene mountain landscape", "A futuristic city at night"]
    pipeline = ImageGenerationPipeline(vote_threshold=3)
    pipeline.generate_initial_sets(prompts)
    while True:
        selected = pipeline.display_and_vote()
        if not selected:
            print("No image reached threshold or quitting.")
            sys.exit()
        variations = [prompts[(selected-1)//3] for _ in range(3)]
        pipeline.iterate(selected, variations)