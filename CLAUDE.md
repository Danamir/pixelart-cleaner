# Pixel Art Cleaner

## Project Overview

Python scripts to convert AI-generated pixel-art images into true pixel-art images by detecting the virtual pixel grid and downscaling to the real pixel-art resolution.

### Problem Statement

AI-generated pixel art is approximative: colors within a single "virtual pixel" (a rectangular block made of many actual pixels) vary slightly, and the virtual pixel grid is not aligned to a perfect grid. Virtual pixel sizes can also vary — some pixels may be wider or taller than others due to AI generation artifacts. The tool must:

1. Detect virtual pixel boundaries by analyzing luma discontinuities between adjacent rows/columns
2. Classify the grid as regular (uniform virtual pixel size) or irregular (variable sizes)
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
venv/Scripts/python.exe detect.py <input> [options]

# Downsampling (main tool)
venv/Scripts/python.exe resize.py <input>... [options]
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
├── detect.py        # Grid detection + edge overlay output
├── resize.py        # Downsampling using detected grid
└── venv/            # Virtual environment (not committed)
```

Test images live in `test.local/` (git-ignored via `*.local` pattern).

---

## Algorithm — detect.py

### Step 1: Luma conversion (`to_luma`)

Converts the RGB image to a single-channel luma image using Rec.601 weights:
`L = 0.299·R + 0.587·G + 0.114·B`

All boundary detection operates on luma rather than RGB to reduce noise from color variation within a virtual pixel.

### Step 2: Per line-pair fractions (`compute_break_fractions`)

For each pair of adjacent rows (or columns), computes the fraction of perpendicular positions where the absolute luma difference exceeds a threshold.

Two robustness measures:

1. **Minimum absolute threshold** (`--min-edge`, default 10): the effective threshold is `max(percentile_threshold, min_luma_diff)`. Prevents near-zero thresholds on images with flat or white/black backgrounds where the percentile of all diffs collapses to noise.

2. **Active-position normalization**: the fraction is computed only over perpendicular positions that fire above threshold at least once anywhere in the image. Prevents a small sprite on a large blank background from being diluted by the inactive background columns/rows.

### Step 3: Peak detection (`find_breaks_in_profile`)

Finds peaks in the fraction profile using `scipy.signal.find_peaks`. The adaptive height threshold is `max(5% of profile max, 1.5× profile mean)`. Minimum distance between peaks is controlled by `--min-distance` (default 3 px).

### Step 4: Band-based voting (`detect_breaks_banded`)

Rather than running detection once over the full image, splits the image into 64 px-wide perpendicular bands and detects breaks independently per band. Positions are collected into a vote counter; positions appearing in at least `min_votes` bands (default 1) survive.

Nearby surviving positions within `--cluster-radius` px (default 1) are merged into a single break using a vote-weighted mean. This approach makes each band's normalization local, so a feature spanning only a few rows/columns still produces full-strength peaks within its bands instead of being diluted globally.

### Step 5: Regularity analysis (`analyze_regularity`)

Converts break positions to span sizes (pixel distances between consecutive breaks including image edges), then:

1. Computes median span size.
2. Filters out spans > 3× median (wide art gaps that are not virtual pixel boundaries).
3. Re-computes median on the filtered set.
4. Computes the coefficient of variation `CV = std / median` on the filtered spans. If `CV ≤ --regular-tolerance` (default 0.20), the axis is classified as **regular**.

Returns `(is_regular, median_size)`.

### Step 6: Break refinement

Only applied when a valid period is found (median span > 2 px):

- `remove_close_breaks`: drops duplicate detections closer than `0.5 × period`.
- `fill_missing_breaks`: inserts synthetic breaks in spans wider than `--max-gap × period` (default 1.6). Handles leading, internal, and trailing gaps. Synthetic breaks are tracked separately for overlay rendering.

### Detection output (`detect_pixel_grid_v3`)

Returns 8 values:

| Value | Description |
|---|---|
| `pixel_w` | Median virtual pixel width in px (None if undetectable) |
| `pixel_h` | Median virtual pixel height in px (None if undetectable) |
| `is_regular_w` | True if column grid is regular |
| `is_regular_h` | True if row grid is regular |
| `col_breaks` | All column break positions |
| `row_breaks` | All row break positions |
| `col_synth` | Column breaks inserted by gap-filling |
| `row_synth` | Row breaks inserted by gap-filling |

### Edge overlay output

- **Color overlay** (default): detected breaks drawn over the original image. True breaks are solid lines (red = rows, blue = columns). Synthetic (gap-filled) breaks are drawn as dotted lines at lower alpha.
- **Edge-only** (`--edge-only`): black background with white solid lines for true breaks, grey dotted lines for synthetic breaks.

---

## Algorithm — resize.py

### Grid strategy selection

After detection, the strategy is chosen:

| Condition | Strategy |
|---|---|
| Both axes regular, no override | **Regular** |
| Either axis irregular, or `--force-irregular` | **Irregular** |
| `--force-regular` | **Regular** (uses period even if irregular) |

### Regular strategy

Uses `regular_grid_spans`: fits a phase offset to the detected break positions via circular mean, then places uniformly spaced spans across the full image at the detected period. This correctly handles sub-pixel grid drift.

With `--square`: averages `pixel_w` and `pixel_h` into a single square period before placing spans.

### Irregular strategy

Uses `spans_from_breaks`: converts break positions directly to `(start, end)` spans covering the full image. Each span is sampled independently.

With `--square` (`downsample_square_irregular`): each span is sampled once, then its color is repeated `round(span_size / target_size)` times, where `target_size = min(pixel_w, pixel_h)`. Wide spans produce multiple adjacent identical output pixels rather than being sub-sampled.

### Sampling methods (`--sample`)

| Method | Description |
|---|---|
| `center` (default) | Single center pixel of the block |
| `center_region` | Mean of the inner 50% (trims 25% each edge to avoid anti-aliased boundaries) |
| `max` | Mean of pixels in the most common color bucket (quantizes each channel into 10 levels, finds the dominant bucket, averages actual unquantized pixels in it) |

### Tiled mode (`--tile SIZE`)

Splits the image into tiles (e.g. `64x64`) and runs full detection independently per tile. Useful when virtual pixel sizes vary across the image.

The `_edges.png` verbose output in tiled mode shows per-tile detected breaks translated to global coordinates, with green lines at alpha 0.5 marking tile boundaries.

Note: tiled mode is best suited for irregular grids. Output tiles are assembled with max-height/max-width padding per strip.

---

## CLI Options Reference

### detect.py

| Option | Short | Default | Description |
|---|---|---|---|
| `--output=PATH` | `-o` | `<input>_edges.png` | Output path |
| `--edge-percentile=P` | `-p` | 85 | Luma-diff percentile threshold; raise to ignore weaker boundaries |
| `--palette-colors=N` | | 256 | Max colors for palette count report |
| `--max-gap=R` | `-g` | 1.6 | Max gap ratio before gap-filling subdivides a span |
| `--regular-tolerance=T` | `-t` | 0.20 | Max CV (std/median) of span sizes for regularity; raise to be more permissive |
| `--min-edge=E` | `-e` | 10 | Minimum absolute luma diff to count as an edge; prevents collapse on flat backgrounds |
| `--min-distance=D` | `-d` | 3 | Minimum px between two break peaks; lower to detect closely-spaced boundaries |
| `--cluster-radius=C` | `-c` | 1 | Max px to merge nearby band votes into one break; lower to preserve close breaks |
| `--edge-only` | | off | Output greyscale grid instead of color overlay |

### resize.py

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
| `--sample=METHOD` | `-m` | center | Sampling method: `center`, `center_region`, `max` |
| `--square` | `-q` | off | Force square output pixels |
| `--verbose` | `-v` | off | Also save `_edges.png`, `_compare.png`, `_compare_true.png` |

#### Verbose output files

| File | Description |
|---|---|
| `<stem>_pixel.png` | Downsampled pixel-art output (always saved) |
| `<stem>_edges.png` | Detected grid overlaid on source; in tiled mode shows per-tile breaks + green tile boundaries |
| `<stem>_compare.png` | Output scaled back to original dimensions (nearest-neighbour) |
| `<stem>_compare_true.png` | Output scaled up by largest integer factor that fits |

Verbose filenames encode active non-default options, e.g. `ship_i_q_g3_p90_pixel.png`.

---

## --square option in detail

`--square` / `-q` behavior differs by strategy:

**Regular mode**: averages `pixel_w` and `pixel_h` into a single square period before placing grid centers.

**Irregular mode** (`downsample_square_irregular`): each detected span is sampled **once**. Its color is then repeated `round(span_size / target_size)` times along its axis, where `target_size = min(pixel_w, pixel_h)`. This means:
- A normal span (~`target_size` px) → 1 output pixel
- A wide/tall span (2× target) → 2 identical adjacent output pixels
- A very wide span preserved by `--max-gap` (3× target) → 3 identical output pixels

No independent sub-sampling of sub-regions is done, so no spurious edge artefacts appear within a single virtual pixel.

**Recommended combination for images with wide irregular pixels**:
```bash
venv/Scripts/python.exe resize.py image.png -i -q -g 3.0
```

---

## Debug

- File `test.local/test.png`, 64x64 image with 5x5 virtual pixels. Should register 8 true rows, 6 true cols:
```

    ##
   #xx#
  #xxxx#
  #xxxx#
   #xx#
    ##

  xxxxxx


```

- File `test.local/test-line.png`, 1152x36 image extracted from an AI generated one, with approximate 5x5 virtual pixels. Should register 4 true rows, 8 true cols:
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
| Too few breaks / boundaries missed | Threshold too high or min-edge too high | Lower `-p` or `-e` |
| Wide pixels over-subdivided | `--max-gap` too low | Raise `-g` (e.g. 2.0–4.0) |
| Closely-spaced breaks merged into one | `--cluster-radius` too high | Lower `-c` to 1 |
| Two adjacent breaks detected as one | `--min-distance` too high | Lower `-d` |
| Output squished (rectangular pixels) | Asymmetric virtual pixel detection | Add `-q` |
| `-q` with wrong pixel count | Regular mode selected but pixels are irregular | Add `-i` to force irregular + `-q` |
| Output colors washed / blended | Anti-aliased virtual-pixel boundaries sampled | Use `-m center` or `-m max` instead of `center_region` |
| Small feature breaks not detected | Feature too narrow for global normalization | Already handled by band-based detection; try lowering `-p` |
| Breaks only partially correct | Mixed regular/irregular content | Use `-i` to rely on raw detected spans |
