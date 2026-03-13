# Pixel Art Cleaner

Licensed under the [GNU General Public License v3.0](LICENCE).

Convert AI-generated pixel-art images into true pixel art by detecting the virtual pixel grid and downscaling to the real pixel-art resolution.

AI-generated pixel art is approximative: colors within a single "virtual pixel" vary slightly, and the grid is not perfectly aligned. This tool detects virtual pixel boundaries from luma discontinuities, then downsamples each block into a single output pixel.

## Installation

Requires Python 3.9+ and [uv](https://github.com/astral-sh/uv).

```bash
# Create and activate a virtual environment
uv venv
. venv/Scripts/activate   # Windows (Git Bash)
# source venv/bin/activate  # Linux / macOS

# Install dependencies and register CLI commands
uv pip install -e .
```

## Usage

```bash
# Activate the environment first
. venv/Scripts/activate

# Detect the virtual pixel grid (produces an annotated overlay image)
detect <input> [options]

# Downsample to true pixel-art resolution
resize <input>... [options]
```

Output is always PNG regardless of input format.

### Quick examples

```bash
# Basic downsampling
resize sprite.png

# Relaxed gap tolerance for wide irregular pixels (square pixels on by default)
resize sprite.png -g 3.0

# Inspect the detected grid before downsampling
detect sprite.png -o sprite_grid.png

# Verbose output: also save edge overlay, upscaled compare images
resize sprite.png -v
```

---

## CLI Options Reference

### detect

Detects the virtual pixel grid and writes an annotated overlay image.

```
detect <input> [options]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--output=PATH` | `-o` | `<input>_edges.png` | Output path |
| `--edge-percentile=P` | `-p` | 85 | Luma-diff percentile threshold; raise to ignore weaker boundaries |
| `--palette-colors=N` | | 256 | Max colors for palette count report |
| `--max-gap=R` | `-g` | 1.6 | Max gap ratio before gap-filling subdivides a span |
| `--regular-tolerance=T` | `-t` | 0.20 | Max CV (std/median) of span sizes for regularity; raise to be more permissive |
| `--min-edge=E` | `-e` | 10 | Minimum absolute luma diff to count as an edge; prevents collapse on flat backgrounds |
| `--min-distance=D` | `-d` | 3 | Minimum px between two break peaks; lower to detect closely-spaced boundaries |
| `--cluster-radius=C` | `-c` | 1 | Max px to merge nearby band votes into one break |
| `--edge-only` | | off | Output greyscale grid instead of color overlay |

### resize

Detects the grid and downsamples to true pixel-art resolution.

```
resize <input>... [options]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--output=PATH` | `-o` | `<stem>_pixel.png` | Output path |
| `--edge-percentile=P` | `-p` | 85 | Luma-diff percentile threshold |
| `--min-edge=E` | `-e` | 10 | Minimum absolute luma diff floor |
| `--min-distance=D` | `-d` | 3 | Minimum px between break peaks |
| `--cluster-radius=C` | `-c` | 1 | Max px to merge band votes |
| `--max-gap=R` | `-g` | 1.6 | Max gap ratio before gap-filling |
| `--regular-tolerance=T` | | 0.20 | Max CV (std/median) of span sizes for regularity |
| `--force-regular` | `-r` | off | Always use regular grid strategy |
| `--force-irregular` | `-i` | off | Always use irregular span strategy |
| `--scale=S` | `-s` | 1.0 | Divide detected pixel size by S (S>1 → finer grid) |
| `--tile=SIZE` | `-t` | off | Process in independent tiles, e.g. `64x64` |
| `--sample=METHOD` | `-m` | `center` | Sampling method: `center`, `center_region`, `max` |
| `--no-square` | `-q` | off | Disable square output pixels (on by default) |
| `--verbose` | `-v` | off | Also save `_edges.png`, `_compare.png`, `_compare_true.png` |

#### Sampling methods

| Method | Description |
|---|---|
| `center` (default) | Single center pixel of the block |
| `center_region` | Mean of the inner 50% (trims 25% each edge to avoid anti-aliased boundaries) |
| `max` | Mean of pixels in the most common color bucket |

#### Verbose output files

| File | Description |
|---|---|
| `<stem>_pixel.png` | Downsampled pixel-art output (always saved) |
| `<stem>_edges.png` | Detected grid overlaid on source |
| `<stem>_compare.png` | Output scaled back to original dimensions (nearest-neighbour) |
| `<stem>_compare_true.png` | Output scaled up by largest integer factor that fits |

Verbose filenames encode active non-default options, e.g. `ship_i_q_g3_p90_pixel.png`.

---

## Square pixels (`--no-square` / `-q`)

Square output pixels are **on by default**. Pass `-q` / `--no-square` to disable.

The default strategy is always **irregular** (raw detected spans). If a regular grid is detected, `resize` will print a hint suggesting `--force-regular` (`-r`) for potentially better results.

**Irregular mode** (default): each detected span is sampled once. Its color is repeated `round(span_size / target_size)` times along its axis, where `target_size = min(pixel_w, pixel_h)`. This means:

- A normal span (~`target_size` px) → 1 output pixel
- A wide/tall span (2× target) → 2 identical adjacent output pixels
- A very wide span (3× target) → 3 identical output pixels

No independent sub-sampling is done within a span, so no spurious edge artefacts appear inside a single virtual pixel.

**Regular mode** (`-r`): averages the detected `pixel_w` and `pixel_h` into a single square period before placing grid centers.

**Recommended combination for images with wide irregular pixels:**
```bash
resize image.png -g 3.0
```

**To disable square pixels:**
```bash
resize image.png -q
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| Too many breaks detected | Threshold too low | Raise `-p` (e.g. 90, 95) |
| Too few breaks / boundaries missed | Threshold too high or min-edge too high | Lower `-p` or `-e` |
| Wide pixels over-subdivided | `--max-gap` too low | Raise `-g` (e.g. 2.0–4.0) |
| Closely-spaced breaks merged into one | `--cluster-radius` too high | Lower `-c` to 1 |
| Two adjacent breaks detected as one | `--min-distance` too high | Lower `-d` |
| Output squished (rectangular pixels) | Square mode disabled | Remove `-q` (square is on by default) |
| Square output has wrong pixel count | Irregular spans with very wide pixels | Raise `-g` (e.g. 2.0–4.0) |
| Output colors washed / blended | Anti-aliased boundaries sampled | Use `-m center` or `-m max` instead of `center_region` |
| Breaks only partially correct | Mixed regular/irregular content | Use `-i` to rely on raw detected spans |
