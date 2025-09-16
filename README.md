# Aesthetic Selection (SDXL crowd-driven evolution)

Iteratively evolve images with Stable Diffusion XL (SDXL) using live audience voting. The workflow generates small batches of images, displays them as a numbered gallery, collects votes via keyboard, and once a vote threshold is reached, uses the winning image as the seed for the next generation (img2img). Over time this introduces selection pressure into a stochastic generative process.

Two small scripts work together:

- `aestheticSelectionSDXL.py` — main pipeline (generation, voting loop, saving galleries)
- `gallery_viewer.py` — live viewer that auto-refreshes to show the latest gallery

Works on single GPU, multi-GPU (optional sharding via Accelerate), and CPU (very slow).

The gallery viewer is optional for the core loop but recommended for interactive demos.

## How it works

1. You enter a vote threshold and three base prompts with an initial guidance scale.
2. The pipeline generates 3 images per prompt (total 9), saves them under `outputs/`, and writes a stitched gallery image `gallery_gen_0.png` with numbers 1–9 overlaid.
3. In the console, attendees vote using number keys 1–9. When an image within a prompt set reaches the vote threshold, that image becomes the init image for SDXL img2img and three new variations are created for that “set”.
4. A new gallery is saved (`gallery_gen_1.png`, `gallery_gen_2.png`, …) and voting continues indefinitely until you quit.

Voting notes:
- Keys: `1–9` to vote, `q` to quit.
- Votes reset only for the mutated set after each iteration; other sets retain their current tallies.


## Requirements

- Python 3.10+ (recommended)
- NVIDIA GPU with CUDA for reasonable performance (built for HPC so VRAM requirements are demanding). CPU mode is supported but painfully slow.
- Internet access for first run to download SDXL weights from Hugging Face.

Python packages:
- diffusers, transformers, accelerate, safetensors
- torch (install per your CUDA/OS), xformers (optional, improves memory and speed on CUDA)
- Pillow, matplotlib

See `requirements.txt` for a base list. For PyTorch, follow the official instructions below.


## Installation

It’s best to use a fresh virtual environment.

1) Create and activate an environment

Conda example (Windows PowerShell):

```powershell
conda create -n aesthetic python=3.10 -y; conda activate aesthetic
```

venv example:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2) Install PyTorch

Install the correct wheel from https://pytorch.org/ matching your CUDA version. Example for CUDA 12.1 on Windows:

```powershell
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

CPU-only (works but slow):

```powershell
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3) Install the rest of the dependencies

```powershell
pip install -r requirements.txt
```

Notes:
- xFormers is optional. If it fails to install, the pipeline falls back automatically. You can skip it and still run.
- Multi-GPU sharding uses Accelerate if available; otherwise the script runs on a single device.


## Running the pipeline and live viewer

Open two terminals in the repo root.

Terminal A — run the SDXL pipeline and voting loop:

```powershell
python aestheticSelectionSDXL.py
```

You’ll be prompted for:
- Vote-winning threshold (integer)
- Three base prompts (text)
- Initial guidance scale (float, typical 5–10)
- Whether to include the initial prompt during img2img (y/n)
- img2img strength (0–1; higher = bigger changes)
- img2img guidance scale (float, typical 5–10)

The script will generate initial images, save numbered galleries in `outputs/`, and then wait for votes in the console. Type numbers 1–9 to vote; press `q` to quit.

Terminal B — run the live gallery viewer:

```powershell
python gallery_viewer.py
```

This opens a matplotlib window that refreshes automatically when a new `gallery_gen_*.png` appears in `outputs/`. It displays the generation number in the window title.


## Outputs

- `outputs/set_1/`, `outputs/set_2/`, `outputs/set_3/` — per-prompt images and iteration folders
- `outputs/set_{k}_gen_{N}/` — img2img variations generated when set k wins at generation N
- `outputs/gallery_gen_0.png`, `outputs/gallery_gen_1.png`, … — stitched galleries with overlaid indices for voting


## Performance and memory tips

- GPU VRAM: SDXL benefits from 12–16GB VRAM. On smaller GPUs, expect slower speed. The script enables attention slicing on single-device runs to reduce peaks.
- xFormers: If installed, it’s enabled automatically for memory-efficient attention.
- Multi-GPU: If you have multiple CUDA devices and `accelerate` is installed, the script attempts to shard the model across GPUs.
- img2img strength: Lower values (e.g., 0.3–0.6) preserve more of the source image; higher values (e.g., 0.8–0.95) explore more.
- Including the original prompt in img2img keeps the images "anchored" on theme. I recommend putting strength very high (>0.9) if retaining the original prompt and keeping strength lower (<0.8) if excluding the original prompt.
- Guidance scale: Typical range 5–10. Higher guidance follows the prompt more strongly, lower guidance is more free-form.


## Troubleshooting

- Module not found (diffusers/transformers/etc.): Ensure your environment is active and `pip install -r requirements.txt` completed successfully.
- torch or CUDA issues on Windows: Install PyTorch from the official index URL matching your CUDA. See the “Install PyTorch” step.
- xFormers failed to enable: Safe to ignore; the script continues without it.
- Out of memory (OOM): Try reducing inference steps, using lower-strength img2img, or running fewer parallel operations. The script already turns on attention slicing for single-device runs.
- Fonts for index labels: The script tries common fonts like DejaVuSans/Arial and falls back to a default. Place a TTF like `DejaVuSans.ttf` in the working directory for consistent labeling.


## License

See `LICENSE` for details.


## Acknowledgments

Built on top of Hugging Face Diffusers and SDXL by Stability AI. Thank you to [Sam Friedman](github.com/samburger) for integrating the gallery_viewer.py with HPC. Thank you to the AI at Yale Symposium for accepting this project as an exhibition.
