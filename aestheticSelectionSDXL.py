"""
This script uses Stable Diffusion XL (SDXL) to generate and iteratively mutate images based on votes.
It can run on single- or multi-GPU systems and accepts command-line parameters for flexibility.
"""

import os
import sys
from collections import Counter

import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

# First, we define a class for the SD pipelines and their parameters
class ImageGenerationPipeline:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vote_threshold: int = 3
    ):

        # Establish device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load pre-trained SDXL text-to-image (initial image set generation)
        self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        # Load pre-trained SDXL img2img (iterate upon the vote winner)
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        # Check number of GPUs available. If > 1, try to shard/parallelize
        num_gpus = torch.cuda.device_count()
        if self.device == "cuda" and num_gpus > 1:
            try:
                from accelerate import infer_auto_device_map, dispatch_model  # type: ignore

                max_mem = {}
                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    # take 90% of total as a safety margin
                    usable_bytes = int(props.total_memory * 0.9)
                    usable_gb = usable_bytes / (1024**3)
                    max_mem[i] = f"{usable_gb:.2f}GB"

                print(f"Sharding across {num_gpus} GPUs with max_memory={max_mem}")

                def shard(module):
                    device_map = infer_auto_device_map(module, max_memory=max_mem)
                    return dispatch_model(module, device_map=device_map)

                # Shard SDXL txt2img
                self.txt2img_pipe.unet = shard(self.txt2img_pipe.unet)
                self.txt2img_pipe.vae = shard(self.txt2img_pipe.vae)
                self.txt2img_pipe.text_encoder = shard(self.txt2img_pipe.text_encoder)
                self.txt2img_pipe.text_encoder_2 = shard(self.txt2img_pipe.text_encoder_2)

                # Shard SDXL img2img
                self.img2img_pipe.unet = shard(self.img2img_pipe.unet)
                self.img2img_pipe.vae = shard(self.img2img_pipe.vae)
                self.img2img_pipe.text_encoder = shard(self.img2img_pipe.text_encoder)
                self.img2img_pipe.text_encoder_2 = shard(self.img2img_pipe.text_encoder_2)
            except Exception as e:
                print(f"Multi-GPU sharding unavailable ({e}). Falling back to single-device execution.")
                self.txt2img_pipe.to(self.device)
                self.img2img_pipe.to(self.device)
        else:
            if self.device == "cuda":
                print("Single GPU detected—running without sharding.")
                self.txt2img_pipe.to(self.device)
                self.img2img_pipe.to(self.device)
            else:
                print("CUDA not available—running on CPU. This will be slow.")

            # Slicing is slower but reduces max VRAM substantially
            self.txt2img_pipe.enable_attention_slicing()
            self.img2img_pipe.enable_attention_slicing()

        # Enable memory-efficient attention for both pipelines when possible
        if self.device == "cuda":
            try:
                self.txt2img_pipe.enable_xformers_memory_efficient_attention()
                self.img2img_pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"xFormers not available or failed to enable ({e}). Continuing without it.")

        # Sets the number of votes needed for a winner to command-line input
        self.vote_threshold = vote_threshold
        self.image_sets = []

        # Create an iteration/generation counter
        self.generation = 0

        # Track votes across rounds per image index
        self.votes = Counter()

    # We generate three initial images for each prompt
    def generate_initial_sets(
        self,
        prompts: list[str],
        num_variations: int = 3, # overwritten by user input
        output_dir: str = "outputs",
        guidance_scale: float = 5, # overwritten by user input
        num_inference_steps: int = 50 # arbitrary, but good balance
    ) -> list[list[str]]:
        os.makedirs(output_dir, exist_ok=True)
        self.image_sets = []
        
        for idx, prompt in enumerate(prompts):
            set_dir = os.path.join(output_dir, f"set_{idx+1}")
            os.makedirs(set_dir, exist_ok=True)
            paths = []

            for j in range(num_variations):
                out = self.txt2img_pipe(
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
        gallery_name: str | None = None
    ):

        if gallery_name is None:
            gallery_name = f"gallery_gen_{self.generation}.png"  # dynamic naming

        rows = []
        # Annotate images with their vote key numbers
        for set_idx, paths in enumerate(self.image_sets):
            imgs = []
            for img_idx, p in enumerate(paths):
                im = Image.open(p).convert("RGB")
                draw = ImageDraw.Draw(im)

                # calculate flat image index (1-based)
                key_num = set_idx * len(paths) + img_idx + 1

                # Try a few common fonts; otherwise fall back to PIL default
                font = None
                for font_name in [
                    "DejaVuSans.ttf",
                    "Arial.ttf",
                    "arial.ttf",
                    "LiberationSans-Regular.ttf",
                ]:
                    try:
                        font = ImageFont.truetype(font_name, size=32)
                        break
                    except Exception:
                        font = None
                if font is None:
                    font = ImageFont.load_default()

                # Draw key number in the top-left corner
                draw.text((8, 8), str(key_num), fill=(255, 255, 255), font=font)

                imgs.append(im)

            # stitch row
            widths, heights = zip(*(im.size for im in imgs))
            total_width = sum(widths)
            max_height = max(heights)
            row_img = Image.new("RGB", (total_width, max_height))
            x = 0
            for im in imgs:
                row_img.paste(im, (x, 0))
                x += im.width
            rows.append(row_img)

        if not rows:
            return

        # combine rows into full gallery
        w = rows[0].width
        h = sum(r.height for r in rows)
        gallery = Image.new("RGB", (w, h))
        y = 0
        for r in rows:
            gallery.paste(r, (0, y))
            y += r.height

        # save annotated gallery
        path = os.path.join(output_dir, gallery_name)
        gallery.save(path)
        print(f"Gallery saved to {path}")

    def display_and_vote(self) -> int | None:
        print("Vote: keys 1–9 (q to quit).")
        total = len(self.image_sets) * len(self.image_sets[0])

        while True:
            print("Current vote counts:")
            for idx in range(1, total + 1):
                print(f"  Image {idx}: {self.votes[idx]} votes")

            k = input("Your vote: ").strip().lower()
            if k == "q":
                return None
            if k.isdigit() and 1 <= (i := int(k)) <= total:
                self.votes[i] += 1
                print(f"  ✓ image {i} → {self.votes[i]} votes")
                if self.votes[i] >= self.vote_threshold:
                    return i
            else:
                print(f"  ✗ Enter 1–{total} or q")

    def iterate(
        self,
        selected_key: int,
        prompts_variation: list[str],
        num_variations: int = 3,
        output_dir: str = "outputs",
        strength: float = 0.9, # overwritten by user input
        guidance_scale: float = 0, # overwritten by user input
        num_inference_steps: int = 70 # arbitrary, higher than initial because reduced with strength
    ) -> list[str]:
        # Derive current number of variations from the first set to avoid mismatch
        current_variations = len(self.image_sets[0]) if self.image_sets else num_variations

        set_i = (selected_key - 1) // current_variations
        img_i = (selected_key - 1) % current_variations
        src = self.image_sets[set_i][img_i]

        init_img = Image.open(src).convert("RGB")

        self.generation += 1  # increment the generation counter
        gen_dir = os.path.join(output_dir, f"set_{set_i+1}_gen_{self.generation}")
        os.makedirs(gen_dir, exist_ok=True)

        new_paths = []
        for j, prompt in enumerate(prompts_variation):
            out = self.img2img_pipe(
                prompt=prompt,
                image=init_img,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            img = out.images[0]
            fn = f"img_{j+1}.png"
            path = os.path.join(gen_dir, fn)
            img.save(path)
            new_paths.append(path)

        self.image_sets[set_i] = new_paths

        # Reset votes for the mutated set only
        for j in range(current_variations):
            flat_idx = set_i * current_variations + j + 1
            self.votes[flat_idx] = 0

        # Save new gallery with iterated images
        self.save_gallery(output_dir, f"gallery_gen_{self.generation}.png")
        return new_paths


if __name__ == "__main__":

    # Prompt user for parameters via command line
    vote_threshold = int(input("Enter vote-winning threshold: ").strip())

    print("\nEnter the prompts for the initial image set generation, along with the guidance scale...\n")
    prompt1 = input("Enter base prompt 1: ").strip()
    prompt2 = input("Enter base prompt 2: ").strip()
    prompt3 = input("Enter base prompt 3: ").strip()
    init_guidance = float(input("Enter initial guidance scale (usually 5-10): ").strip())
  
    print("\nEnter the parameters for img2img (iterations on vote-winner)...\n")
    use_prompt = input("Include initial prompt in img2img? (y/n): ").strip().lower() == 'y'
    iter_strength = float(input("Enter img2img strength (0 no change, 1 max change): ").strip())
    iter_guidance = float(input("Enter img2img guidance scale (usually 5-10): ").strip())


    base_prompts = [prompt1, prompt2, prompt3]
    pipeline = ImageGenerationPipeline(vote_threshold=vote_threshold)
    pipeline.generate_initial_sets(
        base_prompts,
        guidance_scale=init_guidance
    )

    while True:
        sel = pipeline.display_and_vote()
        if sel is None:
            print("Goodbye!")
            sys.exit()

        # Derive current variations per set from pipeline state (fallback to 3)
        current_variations = len(pipeline.image_sets[0]) if pipeline.image_sets else 3
        idx = (sel - 1) // current_variations
        parent_prompt = base_prompts[idx]

        # Decide whether to use the original prompt or an empty prompt
        if use_prompt:
            variations = [f"{parent_prompt}." for _ in range(current_variations)]
        else:
            variations = ["" for _ in range(current_variations)]

        pipeline.iterate(
            sel,
            variations,
            strength=iter_strength,
            guidance_scale=iter_guidance,
        )
