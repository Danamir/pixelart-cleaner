# Pixel Art Cleaner

## Project Overview

A Python script to convert AI-generated pixel-art images into true pixel-art images by detecting the virtual pixel grid and downscaling to the real pixel-art resolution.

### Problem Statement

AI-generated pixel art is approximative: colors within a single "virtual pixel" (a rectangular block made of many actual pixels) vary slightly, and the virtual pixel grid is generally not aligned to a perfect grid. The tool must:

1. Detect virtual pixel boundaries by analyzing color discontinuities between neighboring regions
2. Identify the irregular/non-aligned virtual pixel grid
3. Downscale the image by sampling each detected virtual pixel (e.g., median/average color)
4. Output a clean, true pixel-art image at the correct reduced resolution

## Environment

- **Python interpreter**: `venv/Scripts/python.exe`
- **Package manager**: `venv/Scripts/pip.exe`
- **Platform**: Windows (MINGW64 / Git Bash shell)

## Conventions

- **Argument parsing**: use [`docopt-ng`](https://github.com/jazzband/docopt-ng). The CLI interface is defined by the module docstring; `docopt(__doc__)` parses it. Do **not** use `argparse`.

## Running the Project

```bash
venv/Scripts/python.exe main.py <input_image> [options]
```

Install dependencies:

```bash
venv/Scripts/pip.exe install -r requirements.txt
```

## Project Structure

```
pixelart-cleaner/
├── CLAUDE.md
├── requirements.txt
├── main.py          # Entry point / CLI
└── venv/            # Virtual environment (not committed)
```

## Key Technical Concepts

- **Virtual pixel**: A rectangular block of actual pixels that represents a single pixel in the original pixel-art intent. Size and position are irregular due to AI generation.
- **Grid detection**: Use edge/gradient analysis or color-change heuristics to detect boundaries between virtual pixels.
- **Sampling strategy**: Once virtual pixels are identified, sample each block (e.g., median color) to produce the output pixel.
