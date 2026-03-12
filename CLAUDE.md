# Pixel Art Cleaner

## Project Overview

Python scripts to convert AI-generated pixel-art images into true pixel-art images by detecting the virtual pixel grid and downscaling to the real pixel-art resolution. Two generations exist: v1 (`detect.py` / `resize.py`) and v2 (`detect2.py` / `resize2.py`) with improved detection algorithms.

### Problem Statement

AI-generated pixel art is approximative: colors within a single "virtual pixel" (a rectangular block made of many actual pixels) vary slightly, and the virtual pixel grid is not aligned to a perfect grid. Virtual pixel sizes can also vary — some pixels may be wider or taller than others due to AI generation artifacts. The tool must:

1. Detect virtual pixel boundaries by analyzing color discontinuities between neighboring regions
2. Estimate the pixel period (regular grid) or enumerate individual spans (irregular grid)
3. Downsample the image by sampling each detected virtual pixel block
4. Output a clean, true pixel-art image at the correct reduced resolution

## Environment

- **Python interpreter**: `venv/Scripts/python.exe`
- **Package manager**: `venv/Scripts/pip.exe`
- **Platform**: Windows (MINGW64 / Git Bash shell)

## Conventions

- **Argument parsing**: use [`docopt-ng`](https://github.com/jazzband/docopt-ng). The CLI interface is defined by the module docstring; `docopt(__doc__)` parses it. Do **not** use `argparse`.
- **Output format**: always PNG, regardless of input format.
- **Short options**: every option that takes a value has a short form; boolean flags use long form only except `-r`, `-i`, `-q`, `-v`.

## Running the Project

```bash
# Grid detection only (produces annotated edge overlay)
venv/Scripts/python.exe detect.py <input> [options]     # v1
venv/Scripts/python.exe detect2.py <input> [options]    # v2 (consensus)

# Downsampling (main tool)
venv/Scripts/python.exe resize.py <input>... [options]  # v1
venv/Scripts/python.exe resize2.py <input>... [options] # v2 (consensus + trimmed sampling)
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
├── detect.py        # v1: Grid detection + edge overlay output
├── resize.py        # v1: Downsampling using detected grid
├── detect2.py       # v2: Improved detection (consensus, adaptive fill, 2D refinement)
├── resize2.py       # v2: Downsampling with v2 detection + trimmed sampling
└── venv/            # Virtual environment (not committed)
```

Test images live in `test.local/` (git-ignored via `*.local` pattern).

---

## Algorithm — detect.py

### Step 1: Gradient computation (`compute_gradients`)

Computes per-pixel color difference magnitudes along each axis:
- `mag_h` (H, W-1): horizontal differences (left↔right)
- `mag_v` (H-1, W): vertical differences (top↔bottom)

### Step 2: Break profile (`compute_break_profile`)

For each row position, counts the fraction of **active columns** (columns that ever fire above the gradient threshold) showing a strong vertical gradient there. Normalizing by active columns — not total image width — prevents sparse sprites on large backgrounds from being diluted.

### Step 3: Peak detection (`find_break_positions`)

Finds peaks in the break profile using `scipy.signal.find_peaks`. The adaptive height threshold is `max(5% of peak, 1.5× mean)`. Minimum distance between peaks: 3 px.

### Step 4: Period estimation (`estimate_pixel_period`)

Sweeps all integer art sizes `n` (meaning `n` virtual pixels per axis). For each `n`, scores how well a regular grid with period `image_length / n` aligns with detected breaks (Gaussian residual over the best phase anchor). Returns `image_length / best_n`, or `None` if no candidate scores above 0.3.

Using integer art sizes avoids sub-harmonic aliasing from FFT/comb approaches and directly yields a physically meaningful result.

### Step 5: Break refinement

- `_remove_close_breaks`: removes duplicate detections closer than `0.5 × period`
- `_fill_missing_breaks`: inserts synthetic breaks into gaps wider than `max_gap_ratio × period` (default 1.6). Handles leading, internal, and trailing edge gaps.

---

## Algorithm — resize.py

### Grid strategy selection

After calling `detect_pixel_grid`, the strategy is chosen:

| Condition | Strategy |
|---|---|
| Period estimated, CV low | **Regular** (default) |
| Period not estimated, or `--force-irregular` | **Irregular** |
| `--force-regular` | **Regular** (errors if period unavailable) |

Coefficient of variation (CV) of break spacings measures regularity; CV < 0.25 = consistent.

### Regular strategy

Computes phase-corrected grid centers via `_regular_centers`: uses detected breaks to find the best-fitting phase offset (Gaussian alignment), then places one center per virtual pixel at half-period intervals. Samples by nearest-neighbour at each center.

### Irregular strategy

Converts detected break positions to `(start, end)` spans covering the full image. Samples each span block independently using the configured method (center, mean, or median).

### Tiled mode (`--tile`)

Two-pass algorithm:
1. **Pass 1**: sample each tile independently (local phase correction for regular; global span attribution for irregular)
2. **Pass 2**: determine canonical output size per strip and pad/assemble tiles

Spans are attributed to tiles by their center pixel to avoid double-counting at boundaries (`_spans_in_tile`).

---

## Algorithm — detect2.py (v2)

Addresses three limitations of v1's detection:
1. Single-threshold fragility (one `--edge-percentile` controls everything)
2. Gap-filling requires a global period (fails entirely in irregular mode)
3. Global 1D projection dilutes breaks that only span part of the image

`detect2.py` imports utility functions from `detect.py` and adds three new stages.

### Step 1: Multi-threshold consensus (`consensus_break_positions`)

Instead of detecting breaks at a single percentile, runs `compute_break_profile` + `find_break_positions` at **7 thresholds** evenly spread over a 30-point percentile range (e.g. 70–100 when center is 85).

Peaks from all thresholds are clustered by proximity (radius 2 px). Each cluster counts how many **distinct thresholds** produced a peak in that neighbourhood. Only clusters with votes from >= 40% of thresholds (minimum 2) survive.

Genuine virtual-pixel boundaries produce strong gradient peaks that persist across many thresholds. Noise and anti-aliasing artefacts only fire at the lowest thresholds and are filtered out.

### Step 2: Period estimation + adaptive gap-filling (`adaptive_fill_breaks`)

Period estimation reuses v1's integer art-size sweep on the (now higher-quality) consensus breaks.

**Key improvement**: when period estimation fails (returns `None`), v1 skips gap-filling entirely. V2 falls back to the **median inter-break spacing** as an approximate period, then runs the same `_fill_missing_breaks` logic. This enables gap-filling in irregular mode where v1 could not.

### Step 3: 2D strip-based refinement (`refine_breaks_2d`)

After independent 1D detection for both axes, v2 cross-validates using the perpendicular axis:

- **Column break refinement**: slices the image into horizontal strips (between row breaks). Within each strip, runs column break detection independently. Positions appearing in >= 2 strips (or >= 1/3 of strips, whichever is smaller) that aren't already detected are added.

- **Row break refinement**: same logic using vertical strips (between column breaks).

This catches breaks that the global 1D projection dilutes — e.g. a vertical boundary that only spans a small sprite on a large uniform background. The per-strip projection concentrates the signal.

### Pipeline order

1. Consensus breaks (both axes)
2. Period estimation (integer art-size sweep)
3. Remove close duplicates (if period known)
4. Adaptive gap-filling (global period or median-spacing fallback)
5. 2D strip-based refinement
6. Final duplicate cleanup

---

## Algorithm — resize2.py (v2)

`resize2.py` monkey-patches `resize.py` at import time, replacing two components:

### Detection patch

Replaces `detect_pixel_grid` with `detect_pixel_grid_v2` in the `resize` module namespace. Because Python resolves module-level names at call time, this affects all code paths including `_downsample_tiled`'s per-tile detection.

### Boundary-trimmed sampling

AI generators often anti-alias virtual-pixel edges, blending colours from neighbouring pixels into a 1–2 px border zone. V2 replaces `_sample_block` with a trimmed version that **excludes the outer ~15% of each block edge** (for blocks >= 5 px) before sampling. This avoids pulling blended boundary colours into the output.

The trim fraction is fixed at 0.15. It applies to all sampling methods (center, mean, median) in irregular mode, and to `_downsample_square_padded`.

All CLI options are identical to `resize.py`.

---

## v1 vs v2: when to use which

| Scenario | Recommendation |
|---|---|
| Well-behaved regular grid (uniform pixel sizes) | v1 and v2 produce similar results; either works |
| Irregular pixel sizes, period estimation fails | v2 — adaptive gap-filling uses median spacing as fallback |
| Sparse sprite on large background | v2 — 2D strip refinement catches diluted breaks |
| Threshold-sensitive image (results vary with `-p`) | v2 — consensus is robust across a range of thresholds |
| Anti-aliased / blurry virtual-pixel boundaries | v2 — trimmed sampling avoids boundary colour bleed |

---

## CLI Options Reference

### detect.py

| Option | Default | Description |
|---|---|---|
| `-o PATH` / `--output` | `<input>_edges.png` | Output path |
| `--edge-percentile=P` | 85 | Gradient percentile threshold for break detection |
| `--palette-colors=N` | 256 | Max colors for palette quantization report |
| `-g R` / `--max-gap=R` | 1.6 | Max gap ratio before subdividing; increase to allow wider virtual pixels |
| `--edge-only` | off | Output black/white grid instead of color overlay |

### detect2.py

Same options as `detect.py`. Default output path is `<input>_edges2.png`. The `--edge-percentile` value serves as the **center** of the consensus range (actual thresholds span center +/- 15 percentile points).

### resize.py

| Option | Short | Default | Description |
|---|---|---|---|
| `--output=PATH` | `-o` | auto | Output path (auto: `<stem>_pixel.png`) |
| `--edge-percentile=P` | `-p` | 85 | Gradient threshold; raise (90–98) to detect fewer, stronger breaks |
| `--sample=METHOD` | | center | Sampling for irregular: `center`, `mean`, `median` |
| `--max-gap=R` | `-g` | 1.6 | Wider gaps accepted as single virtual pixels before gap-fill subdivides them; raise to 2–4 when legitimately wide pixels exist |
| `--force-regular` | `-r` | off | Always use regular grid strategy |
| `--force-irregular` | `-i` | off | Always use irregular span strategy |
| `--scale=S` | `-s` | 1.0 | Divide detected pixel size by S (S>1 → finer grid, S<1 → coarser) |
| `--tile=SIZE` | `-t` | off | Local phase correction per tile (e.g. `64x64`); helps drifting grids |
| `--square` | `-q` | off | Force square output pixels (see below) |
| `--verbose` | `-v` | off | Also save `_edges.png`, `_compare.png`, `_compare_true.png` |

#### Verbose output files

| File | Description |
|---|---|
| `<stem>_pixel.png` | Downsampled pixel-art output (always saved) |
| `<stem>_edges.png` | Effective sampling grid overlaid on source |
| `<stem>_compare.png` | Output scaled back to original dimensions (nearest-neighbour) |
| `<stem>_compare_true.png` | Output scaled up by largest integer factor |

Verbose filenames encode active non-default options, e.g. `ship_pixel_i_q_g3_p90.png`.

### resize2.py

Same CLI options as `resize.py`. Uses v2 detection and boundary-trimmed sampling automatically.

---

## --square option in detail

`--square` / `-q` behavior differs by strategy:

**Regular mode**: averages `pixel_w` and `pixel_h` into a single square period before placing grid centers. Produces a grid with equal horizontal and vertical spacing.

**Irregular mode** (`_downsample_square_padded`): each detected span is sampled **once**. Its color is then repeated `round(span_size / target_size)` times along its axis, where `target_size = min(pixel_w, pixel_h)`. This means:
- A normal span (~`target_size` px) → 1 output pixel
- A wide/tall span (2× target) → 2 identical output pixels adjacent
- A very wide span preserved by `--max-gap` (3× target) → 3 identical output pixels

No independent sub-sampling of sub-regions is done, so no spurious edge artefacts appear within a single virtual pixel. The output image is larger than without `--square` (extra pixels for wider spans), but each output pixel represents an approximately square source region.

**Recommended combination for images with wide irregular pixels**:
```bash
venv/Scripts/python.exe resize.py image.png -i -q -g 3.0
```

---

## Debug

- File `test.local/test.png` , 64x64 image with 5x5 virtual pixels forming this shape. Should register 8 true rows, 6 true cols : 
```

    ##
   #xx#
  #xxxx#
  #xxxx#
   #xx#
    ##

  xxxxxx


```

- File `test.local/test-line.png` , 1152x36 image extacted from an AI generated one, with approximate 5x5 virtual pixels forming this shape. Should register 4 true rows, 8 true cols : 
```

       
          ######!
        ##xxxxxxx!##
     ##!xxxxxxxxxxxx##
    #xxxxxxxxxxxxxxxxx#
```


---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| Too many breaks detected | Threshold too low | Raise `-p` (e.g. 90, 95) |
| Wide pixels not detected / over-subdivided | `--max-gap` too low | Raise `-g` (e.g. 2.0–4.0) |
| Output squished (rectangular pixels) | Asymmetric virtual pixel detection | Add `-q --square` |
| `-q` causes too many edges | Irregular mode subdividing instead of padding | Ensure using `-i` with `-q`; the padded sampler is only active in non-tiled irregular mode |
| Grid phase drifts across image | AI generation skewed the grid locally | Add `-t 64x64` (or similar tile size) |
| Regular grid gives wrong count | Period estimation misled by dominant boundary | Try `-i` to use span detection directly |
| Results change a lot with small `-p` changes | Single-threshold fragility | Use `resize2.py` (consensus is robust across thresholds) |
| Breaks missed on sparse sprites | Global 1D projection dilutes signal | Use `detect2.py` / `resize2.py` (2D strip refinement) |
| Output colours look washed / blended | Anti-aliased virtual-pixel boundaries sampled | Use `resize2.py` (boundary-trimmed sampling) |
