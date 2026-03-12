#!/usr/bin/env python3
"""Detect the virtual pixel grid in an AI-generated pixel-art image.

Compares adjacent rows (or columns) using the luma channel to find where
significant brightness changes occur across the full line pair, marking
virtual pixel boundaries.  Heights/widths between detected breaks are
collected and analysed to determine whether the grid is regular or irregular.

Usage:
    detect.py <input> [--output=PATH] [--edge-percentile=P] [--palette-colors=N]
               [-g R] [-t T] [-e E] [-d D] [-c C] [--edge-only]
    detect.py -h | --help

Arguments:
    <input>                     Input image path.

Options:
    -o PATH --output=PATH       Output image path.
    -p P --edge-percentile=P    Per-pixel luma-diff percentile threshold [default: 85].
    --palette-colors=N          Max colours for palette report [default: 256].
    -g R --max-gap=R            Max gap ratio before subdividing [default: 1.6].
    -t T --regular-tolerance=T  Span-size tolerance for regularity check (0-1) [default: 0.10].
    -e E --min-edge=E           Minimum absolute luma difference to count as an edge (0-255).
                                Prevents near-zero thresholds on flat/sparse backgrounds [default: 10].
    -d D --min-distance=D       Minimum distance in px between two peaks in the break profile.
                                Lower to detect breaks closer together than 3 px [default: 3].
    -c C --cluster-radius=C     Max distance in px to merge nearby band votes into one break.
                                Lower to preserve closely-spaced breaks [default: 1].
    --edge-only                 Output a greyscale grid image instead of a colour overlay.
    -h --help                   Show this screen.
"""

import sys
from collections import Counter
from pathlib import Path
from math import ceil

import numpy as np
from docopt import docopt
from PIL import Image
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Image loading / colour conversion
# ---------------------------------------------------------------------------

def load_image(path: str) -> tuple[Image.Image, np.ndarray]:
    """Load an RGB image and return both the PIL Image and a float32 numpy array."""
    img = Image.open(path).convert("RGB")
    return img, np.array(img, dtype=np.float32)


def to_luma(arr: np.ndarray) -> np.ndarray:
    """Convert RGB float32 (H, W, 3) to Rec.601 luma (H, W), range 0-255."""
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


# ---------------------------------------------------------------------------
# Break detection from luma line comparisons
# ---------------------------------------------------------------------------

def compute_break_fractions(
    L: np.ndarray,
    threshold_percentile: float,
    axis: int,
    min_luma_diff: float = 10.0,
) -> np.ndarray:
    """
    For each pair of adjacent lines along `axis`, compute the fraction of
    *active* perpendicular positions where the luma difference exceeds the
    threshold.

    axis=0 : compare consecutive rows    -> profile length H-1 (row breaks)
    axis=1 : compare consecutive columns -> profile length W-1 (col breaks)

    Two robustness measures are applied:

    1. Minimum absolute threshold (`min_luma_diff`): the effective threshold is
       max(percentile_threshold, min_luma_diff).  This prevents near-zero
       thresholds on images with mostly flat/white backgrounds, where the
       percentile of all diffs can land in the noise floor.

    2. Active-position normalisation: the fraction is computed only over
       perpendicular positions that fire above threshold at least once in the
       image (i.e. are part of sprite content, not flat background).  This
       prevents a small figure on a wide background from having its boundary
       signal diluted by the vast expanse of inactive columns/rows.
    """
    diffs = np.abs(np.diff(L, axis=axis))       # (H-1, W) or (H, W-1)
    threshold = max(float(np.percentile(diffs, threshold_percentile)), min_luma_diff)
    exceed = diffs > threshold
    perp = 1 - axis

    # Active positions: those that fire above threshold at least once
    active = exceed.any(axis=axis)              # shape: (W,) or (H,)
    if not active.any():
        return np.zeros(diffs.shape[axis], dtype=float)

    restricted = np.compress(active, exceed, axis=perp)
    return restricted.mean(axis=perp)           # (H-1,) or (W-1,)


def find_breaks_in_profile(
    fractions: np.ndarray,
    min_distance: int = 3,
) -> np.ndarray:
    """
    Find peak positions in a fraction profile that mark virtual pixel boundaries.

    The adaptive height threshold requires peaks to exceed both 5% of the
    profile maximum and 1.5x the profile mean.
    """
    if fractions.max() < 1e-10:
        return np.array([], dtype=int)
    height = max(fractions.max() * 0.05, fractions.mean() * 1.5)
    peaks, _ = find_peaks(fractions, height=height, distance=min_distance)
    return peaks


def detect_breaks_banded(
    L: np.ndarray,
    threshold_percentile: float,
    axis: int,
    min_luma_diff: float = 10.0,
    band_size: int = 64,
    min_votes: int = 1,
    cluster_radius: int = 2,
    min_distance: int = 3,
) -> np.ndarray:
    """
    Detect break positions by splitting into perpendicular bands and voting.

    For row breaks (axis=0): slices the image into vertical column bands of
    `band_size` px, detects row breaks independently in each band, then
    collects positions that appear in at least `min_votes` bands.

    For col breaks (axis=1): same idea using horizontal row bands.

    Within each band the active-position normalisation is local to that band,
    so a small feature spanning only a few pixels in the perpendicular
    direction produces full-strength peaks — rather than being diluted by the
    rest of the image.  Positions appearing in enough bands are genuine; noise
    confined to a single band doesn't vote enough to survive.

    Nearby positions from different bands are merged (weighted mean by votes).
    """
    H, W = L.shape
    n_band_dim = W if axis == 0 else H

    votes: Counter = Counter()

    for band_start in range(0, n_band_dim, band_size):
        band_end = min(band_start + band_size, n_band_dim)
        band_L = L[:, band_start:band_end] if axis == 0 else L[band_start:band_end, :]

        fracs = compute_break_fractions(band_L, threshold_percentile, axis, min_luma_diff)
        if fracs.max() < 1e-10:
            continue

        peaks = find_breaks_in_profile(fracs, min_distance=min_distance)
        for p in peaks:
            votes[int(p)] += 1

    candidates = sorted(pos for pos, count in votes.items() if count >= min_votes)
    if not candidates:
        return np.array([], dtype=float)

    # Merge nearby positions: weighted mean by vote count
    clusters = []
    group = [candidates[0]]
    group_votes = [votes[candidates[0]]]

    for pos in candidates[1:]:
        if pos - group[-1] <= cluster_radius:
            group.append(pos)
            group_votes.append(votes[pos])
        else:
            total = sum(group_votes)
            clusters.append(sum(p * v for p, v in zip(group, group_votes)) / total)
            group = [pos]
            group_votes = [votes[pos]]

    total = sum(group_votes)
    clusters.append(sum(p * v for p, v in zip(group, group_votes)) / total)

    return np.array(clusters, dtype=float)


# ---------------------------------------------------------------------------
# Span size analysis and regularity
# ---------------------------------------------------------------------------

def breaks_to_span_sizes(breaks: np.ndarray, image_length: int) -> np.ndarray:
    """
    Convert break positions (gradient space, 0..length-2) to span sizes in pixels.

    A break at index b marks the boundary between pixel b and pixel b+1.
    Spans: [0..b0], [b0+1..b1], ..., [bN+1..length-1].
    """
    bounds = [0] + [int(b) + 1 for b in sorted(breaks)] + [image_length]
    return np.array([bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)], dtype=float)


def analyze_regularity(
    span_sizes: np.ndarray,
    tolerance_frac: float = 0.10,
    outlier_ratio: float = 3.0,
    min_close_frac: float = 0.80,
) -> tuple[bool, float]:
    """
    Determine whether detected virtual pixel sizes indicate a regular grid.

    1. Compute median span size.
    2. Filter out spans > outlier_ratio * median (art gaps, not vp boundaries).
    3. Re-compute median on the filtered set.
    4. If >= min_close_frac of filtered spans are within tolerance_frac of
       the new median, the grid is considered regular.

    Returns (is_regular, median_size).
    """
    if len(span_sizes) == 0:
        return False, 0.0

    median = float(np.median(span_sizes))
    if median < 2.0:
        return False, median

    filtered = span_sizes[span_sizes <= outlier_ratio * median]
    if len(filtered) == 0:
        return False, median

    median = float(np.median(filtered))
    close = np.abs(filtered - median) / median <= tolerance_frac
    is_regular = float(close.mean()) >= min_close_frac
    return is_regular, median


# ---------------------------------------------------------------------------
# Break list refinement
# ---------------------------------------------------------------------------

def fill_missing_breaks(
    breaks: np.ndarray,
    period: float,
    image_length: int,
    max_gap_ratio: float = 1.6,
) -> np.ndarray:
    """
    Insert synthetic breaks in spans wider than max_gap_ratio * period.

    Handles leading, internal, and trailing gaps.
    """
    filled = sorted(float(b) for b in breaks)

    def _insert(span_start: float, span_size: float) -> list[float]:
        n = round(span_size / period) - 1
        if n <= 0:
            return []
        step = span_size / (n + 1)
        return [span_start + step * (j + 1) - 1 for j in range(n)]

    if filled:
        leading = filled[0] + 1
        if leading > period * max_gap_ratio:
            filled = _insert(0, leading) + filled

    i = 0
    while i < len(filled) - 1:
        gap = filled[i + 1] - filled[i]
        if gap > period * max_gap_ratio:
            new = _insert(filled[i] + 1, gap)
            if new:
                filled = filled[: i + 1] + new + filled[i + 1 :]
                i += len(new)
        i += 1

    if filled:
        trailing = image_length - (filled[-1] + 1)
        if trailing > period * max_gap_ratio:
            filled = filled + _insert(filled[-1] + 1, trailing)

    return np.array(sorted(filled))


def remove_close_breaks(breaks: np.ndarray, period: float, min_ratio: float = 0.5) -> np.ndarray:
    """Remove break positions closer than min_ratio * period (duplicate detections)."""
    if len(breaks) < 2:
        return breaks
    min_spacing = period * min_ratio
    kept = [float(breaks[0])]
    for b in breaks[1:]:
        if float(b) - kept[-1] >= min_spacing:
            kept.append(float(b))
    return np.array(kept)


# ---------------------------------------------------------------------------
# Combined detection pipeline
# ---------------------------------------------------------------------------

def _synth_breaks(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Return positions in `after` that are not within 1 px of any position in `before`."""
    if len(before) == 0:
        return after.copy()
    return np.array(
        [b for b in after if np.min(np.abs(before - b)) > 1.0],
        dtype=float,
    )


def detect_pixel_grid_v3(
    arr: np.ndarray,
    threshold_percentile: float,
    max_gap_ratio: float = 1.6,
    regular_tolerance: float = 0.10,
    min_luma_diff: float = 10.0,
    min_distance: int = 3,
    cluster_radius: int = 1,
) -> tuple[float | None, float | None, bool, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect the virtual pixel grid using luma line-pair comparisons.

    For each axis:
      1. Compute per-line-pair fraction of pixels exceeding luma-diff threshold.
      2. Find peaks in the fraction profile -> break positions.
      3. Compute span sizes between breaks.
      4. Analyse regularity (median + 80% within tolerance).
      5. If regular, fill missing breaks and remove duplicates.

    Returns
    -------
    pixel_w      : median virtual pixel width  in px (None if undetectable)
    pixel_h      : median virtual pixel height in px (None if undetectable)
    is_regular_w : True if column grid is regular
    is_regular_h : True if row grid is regular
    col_breaks   : all column break positions (gradient space 0..W-2)
    row_breaks   : all row break positions    (gradient space 0..H-2)
    col_synth    : subset of col_breaks that were inserted by gap-filling
    row_synth    : subset of row_breaks that were inserted by gap-filling
    """
    L = to_luma(arr)
    H, W = L.shape

    # Row breaks (compare consecutive rows, banded over columns)
    row_breaks = detect_breaks_banded(L, threshold_percentile, axis=0,
                                      min_luma_diff=min_luma_diff,
                                      cluster_radius=cluster_radius,
                                      min_distance=min_distance)
    row_sizes = breaks_to_span_sizes(row_breaks, H)
    is_regular_h, pixel_h = analyze_regularity(row_sizes, regular_tolerance)
    row_synth = np.array([], dtype=float)
    if pixel_h > 2.0:
        row_breaks = remove_close_breaks(row_breaks, pixel_h)
        row_breaks_filled = fill_missing_breaks(row_breaks, pixel_h, H, max_gap_ratio)
        row_synth = _synth_breaks(row_breaks, row_breaks_filled)
        row_breaks = row_breaks_filled

    # Column breaks (compare consecutive columns, banded over rows)
    col_breaks = detect_breaks_banded(L, threshold_percentile, axis=1,
                                      min_luma_diff=min_luma_diff,
                                      cluster_radius=cluster_radius,
                                      min_distance=min_distance)
    col_sizes = breaks_to_span_sizes(col_breaks, W)
    is_regular_w, pixel_w = analyze_regularity(col_sizes, regular_tolerance)
    col_synth = np.array([], dtype=float)
    if pixel_w > 2.0:
        col_breaks = remove_close_breaks(col_breaks, pixel_w)
        col_breaks_filled = fill_missing_breaks(col_breaks, pixel_w, W, max_gap_ratio)
        col_synth = _synth_breaks(col_breaks, col_breaks_filled)
        col_breaks = col_breaks_filled

    return (
        pixel_w if pixel_w > 0 else None,
        pixel_h if pixel_h > 0 else None,
        is_regular_w,
        is_regular_h,
        col_breaks,
        row_breaks,
        col_synth,
        row_synth,
    )


# ---------------------------------------------------------------------------
# Palette counting
# ---------------------------------------------------------------------------

def count_palette_colors(img: Image.Image, max_colors: int = 256) -> int:
    quantized = img.quantize(
        colors=max_colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    return len(set(quantized.get_flattened_data()))


# ---------------------------------------------------------------------------
# Output images
# ---------------------------------------------------------------------------

_DOT_SIZE = 0  # pixels on / off for dotted synthetic lines


def _dot_mask(length: int) -> np.ndarray:
    """Boolean mask with alternating DOT_SIZE-pixel on/off segments."""
    idx = np.arange(length)
    if _DOT_SIZE <= 0: return idx
    return (idx // _DOT_SIZE) % 2 == 0


def make_grid_overlay(
    original: Image.Image,
    col_breaks: np.ndarray,
    row_breaks: np.ndarray,
    col_synth: np.ndarray | None = None,
    row_synth: np.ndarray | None = None,
    row_color: tuple[int, int, int] = (255, 50, 50),
    col_color: tuple[int, int, int] = (50, 50, 255),
    alpha: float = 0.85,
    synth_alpha: float = 0.33,
) -> Image.Image:
    """
    Draw detected break lines over the original image (red=rows, blue=cols).

    Detected breaks are drawn as solid lines at full `alpha`.
    Synthetic breaks (gap-filled) are drawn as dotted lines at `synth_alpha`.
    """
    out = np.array(original, dtype=np.float32)
    cr = np.array(row_color, dtype=np.float32)
    cc = np.array(col_color, dtype=np.float32)
    H, W = out.shape[:2]

    synth_row_set = set(round(b) for b in (row_synth if row_synth is not None else []))
    synth_col_set = set(round(b) for b in (col_synth if col_synth is not None else []))

    for r in row_breaks:
        r1 = int(r) + 1
        if not (0 <= r1 < H):
            continue
        if round(r) in synth_row_set:
            mask = _dot_mask(W)
            out[r1, mask] = out[r1, mask] * (1 - synth_alpha) + cr * synth_alpha
        else:
            out[r1, :] = out[r1, :] * (1 - alpha) + cr * alpha

    for c in col_breaks:
        c1 = int(c) + 1
        if not (0 <= c1 < W):
            continue
        if round(c) in synth_col_set:
            mask = _dot_mask(H)
            out[mask, c1] = out[mask, c1] * (1 - synth_alpha) + cc * synth_alpha
        else:
            out[:, c1] = out[:, c1] * (1 - alpha) + cc * alpha

    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def make_grid_only(
    original_size: tuple[int, int],
    col_breaks: np.ndarray,
    row_breaks: np.ndarray,
    col_synth: np.ndarray | None = None,
    row_synth: np.ndarray | None = None,
) -> Image.Image:
    """
    Return a black image with grid lines in white.

    Synthetic (gap-filled) breaks are drawn as dotted grey (128).
    """
    H, W = original_size
    arr = np.zeros((H, W), dtype=np.uint8)

    synth_row_set = set(round(b) for b in (row_synth if row_synth is not None else []))
    synth_col_set = set(round(b) for b in (col_synth if col_synth is not None else []))

    for r in row_breaks:
        r1 = int(r) + 1
        if not (0 <= r1 < H):
            continue
        if round(r) in synth_row_set:
            mask = _dot_mask(W)
            arr[r1, mask] = 128
        else:
            arr[r1, :] = 255

    for c in col_breaks:
        c1 = int(c) + 1
        if not (0 <= c1 < W):
            continue
        if round(c) in synth_col_set:
            mask = _dot_mask(H)
            arr[mask, c1] = 128
        else:
            arr[:, c1] = 255

    return Image.fromarray(arr, mode="L")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    W: int,
    H: int,
    pixel_w: float | None,
    pixel_h: float | None,
    is_regular_w: bool,
    is_regular_h: bool,
    col_breaks: np.ndarray,
    row_breaks: np.ndarray,
    col_synth: np.ndarray,
    row_synth: np.ndarray,
    palette_size: int,
) -> None:
    sep = "-" * 46
    print(f"\n{sep}")
    print("  Detection Report (v3)")
    print(sep)
    print(f"  Original resolution    : {W} x {H} px")

    n_row_true = len(row_breaks) - len(row_synth)
    print(f"  Row breaks (true)      : {n_row_true}")
    print(f"  Row breaks (total)     : {len(row_breaks)}  (+{len(row_synth)} forged)")
    if len(row_breaks) > 1:
        sp = np.diff(np.sort(row_breaks.astype(float)))
        sp = sp[sp >= 2]
        if len(sp):
            print(f"  Row spacing (mean/std) : {sp.mean():.1f} +/- {sp.std():.1f} px")
    if pixel_h:
        reg_label = "regular" if is_regular_h else "irregular"
        print(f"  Virtual pixel height   : ~{pixel_h:.3f} px  ({reg_label})")

    n_col_true = len(col_breaks) - len(col_synth)
    print(f"  Col breaks (true)      : {n_col_true}")
    print(f"  Col breaks (total)     : {len(col_breaks)}  (+{len(col_synth)} forged)")
    if len(col_breaks) > 1:
        sp = np.diff(np.sort(col_breaks.astype(float)))
        sp = sp[sp >= 2]
        if len(sp):
            print(f"  Col spacing (mean/std) : {sp.mean():.1f} +/- {sp.std():.1f} px")
    if pixel_w:
        reg_label = "regular" if is_regular_w else "irregular"
        print(f"  Virtual pixel width    : ~{pixel_w:.3f} px  ({reg_label})")

    print(sep)
    if pixel_w and pixel_h:
        art_w = max(1, round(W / pixel_w))
        art_h = max(1, round(H / pixel_h))
        print(f"  Pixel-art resolution   : {art_w} x {art_h} px")
        print(f"  Downsampling ratio     : {art_w / W * 100:.2f}% / {art_h / H * 100:.2f}%")
    else:
        print("  Pixel-art resolution   : unknown")
    print(f"  Colour palette size    : ~{palette_size} colours")
    print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = docopt(__doc__)

    input_path = Path(args["<input>"])
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args["--output"]) if args["--output"] else (
        input_path.parent / (input_path.stem + "_edges.png")
    )
    threshold_percentile = float(args["--edge-percentile"])
    palette_colors = int(args["--palette-colors"])
    max_gap_ratio = float(args["--max-gap"])
    regular_tolerance = float(args["--regular-tolerance"])
    min_luma_diff = float(args["--min-edge"])
    min_distance = int(args["--min-distance"])
    cluster_radius = int(args["--cluster-radius"])
    edge_only = args["--edge-only"]

    print(f"Loading: {input_path}")
    img, arr = load_image(str(input_path))
    H, W = arr.shape[:2]
    print(f"  {W} x {H} px")

    print("Detecting break lines (v3 luma line-pair) ...")
    pixel_w, pixel_h, is_regular_w, is_regular_h, col_breaks, row_breaks, col_synth, row_synth = \
        detect_pixel_grid_v3(arr, threshold_percentile, max_gap_ratio, regular_tolerance,
                             min_luma_diff, min_distance, cluster_radius)

    print("Counting colour palette ...")
    palette_size = count_palette_colors(img, palette_colors)

    print_report(W, H, pixel_w, pixel_h, is_regular_w, is_regular_h,
                 col_breaks, row_breaks, col_synth, row_synth, palette_size)

    if edge_only:
        out_img = make_grid_only((H, W), col_breaks, row_breaks, col_synth, row_synth)
    else:
        out_img = make_grid_overlay(img, col_breaks, row_breaks, col_synth, row_synth)

    print(f"\nSaving: {output_path}")
    out_img.save(str(output_path))
    print("Done.")


if __name__ == "__main__":
    main()
