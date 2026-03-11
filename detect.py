#!/usr/bin/env python3
"""Detect the virtual pixel grid in an AI-generated pixel-art image.

Outputs a grid-line overlay image (original resolution) marking the detected
horizontal and vertical breaks between virtual pixels, and prints detection
statistics:
  - number of detected pixel-row / pixel-column break lines
  - virtual pixel height and width
  - projected pixel-art resolution and downsampling ratio
  - estimated colour palette size

Algorithm
---------
For each axis the detection runs in two steps:

  1. Break profile  -- for each row position r, count the fraction of columns
     that show a significant top-to-bottom colour change at r (and vice-versa
     for columns).  Peaks in this 1-D profile are the detected break lines.

  2. Period recovery -- given the (sparse) detected break positions, sweep all
     integer pixel-art sizes and score each by how well a regular grid with
     that period aligns with the breaks (Gaussian residual over the best phase
     offset).  The integer art size with the best score gives the exact pixel
     period (= image dimension / art size).

     Using integer art sizes avoids the sub-harmonic aliasing that trips up
     free-period FFT / comb approaches, and directly yields a physically
     meaningful output (whole pixel-art pixel count).

Usage:
    detect.py <input> [--output=PATH] [--edge-percentile=P] [--palette-colors=N] [--edge-only]
    detect.py -h | --help

Arguments:
    <input>                 Input image path.

Options:
    -o PATH --output=PATH       Output image path (default: <input>_edges.<ext>).
    --edge-percentile=P         Gradient percentile threshold for break detection, 0-100 [default: 85].
    --palette-colors=N          Max colours for palette quantisation [default: 256].
    --edge-only                 Output a greyscale grid-line image instead of a colour overlay.
    -h --help                   Show this screen.
"""

import sys
from pathlib import Path

from docopt import docopt
from scipy.signal import find_peaks
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> tuple[Image.Image, np.ndarray]:
    """Load an RGB image and return both the PIL Image and a float32 numpy array."""
    img = Image.open(path).convert("RGB")
    return img, np.array(img, dtype=np.float32)


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def compute_gradients(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel colour-gradient magnitudes along each axis.

    Returns
    -------
    mag_h : ndarray, shape (H, W-1)
        Magnitude of the horizontal colour difference (left to right).
    mag_v : ndarray, shape (H-1, W)
        Magnitude of the vertical colour difference (top to bottom).
    """
    grad_h = np.diff(arr, axis=1)          # (H, W-1, 3)
    grad_v = np.diff(arr, axis=0)          # (H-1, W, 3)
    mag_h = np.sqrt((grad_h ** 2).sum(axis=2))
    mag_v = np.sqrt((grad_v ** 2).sum(axis=2))
    return mag_h, mag_v


# ---------------------------------------------------------------------------
# Break-line detection
# ---------------------------------------------------------------------------

def compute_break_profile(
    mag: np.ndarray,
    threshold_percentile: float,
    along_axis: int,
) -> np.ndarray:
    """
    Compute a 1-D break score profile normalised over *active* positions.

    For each position along `along_axis`, the score is the fraction of
    *active* pixels in the perpendicular direction whose gradient magnitude
    exceeds the threshold.

    An active column (for row-break profiles) is any column that fires above
    the threshold at least once anywhere in the image — i.e. a column that
    contains sprite content rather than flat background.  Normalising by
    active positions rather than total image width/height means that a break
    covering only a small sprite on a large uniform background still scores
    near 1.0, instead of being diluted to sprite_width / image_width.

    Parameters
    ----------
    mag : ndarray
        Gradient magnitude array.
        Pass mag_v (H-1, W) with along_axis=1 for row breaks.
        Pass mag_h (H, W-1) with along_axis=0 for column breaks.
    threshold_percentile : float
        Percentile of all values in `mag` used as the detection threshold.
    along_axis : int
        Axis to aggregate over (1 = row breaks from mag_v, 0 = col breaks from mag_h).
    """
    threshold = np.percentile(mag, threshold_percentile)
    high = mag > threshold

    # Active slice mask: perpendicular positions that fire above threshold
    # at least once (i.e. are part of the sprite, not flat background).
    # For along_axis=1 (row profile): active columns = high.any(axis=0)  shape (W,)
    # For along_axis=0 (col profile): active rows    = high.any(axis=1)  shape (H,)
    perp_axis = 1 - along_axis
    active = high.any(axis=perp_axis)

    if not active.any():
        return high.astype(float).mean(axis=along_axis)

    # Restrict to active slices and compute mean along the profile axis
    restricted = np.compress(active, high, axis=along_axis)
    return restricted.astype(float).mean(axis=along_axis)


def find_break_positions(
    profile: np.ndarray,
    min_distance: int = 3,
) -> np.ndarray:
    """
    Find positions of significant colour breaks as peaks in a break profile.

    Returns an array of integer indices into `profile` where peaks occur.
    The height threshold is adaptive: a peak must exceed both 5 % of the
    profile maximum and 1.5 x the profile mean, whichever is larger.
    """
    if profile.max() < 1e-10:
        return np.array([], dtype=int)
    height = max(profile.max() * 0.05, profile.mean() * 1.5)
    peaks, _ = find_peaks(profile, height=height, distance=min_distance)
    return peaks


def _score_art_size(
    positions: np.ndarray,
    image_length: int,
    n: int,
    tolerance: float,
) -> float:
    """
    Score a candidate pixel-art size `n` (meaning `n` virtual pixels along an
    axis of length `image_length`).

    For each detected break position used as phase anchor, compute the
    Gaussian-weighted residual between every detected position and the nearest
    grid line of the regular grid (period = image_length / n).  Return the
    maximum score across all phase anchors.
    """
    p = image_length / n
    best = 0.0
    for anchor in positions:
        phase   = anchor % p
        rem     = (positions - phase) % p
        aligned = np.minimum(rem, p - rem)
        s = float(np.exp(-0.5 * (aligned / tolerance) ** 2).mean())
        if s > best:
            best = s
    return best


def estimate_pixel_period(
    positions: np.ndarray,
    image_length: int,
    tolerance: float = 1.0,
    min_period: float = 3.0,
) -> float | None:
    """
    Estimate the virtual pixel period (in original pixels) from detected break
    positions by sweeping integer pixel-art sizes.

    Pixel art always has integer pixel counts, so the true period is exactly
    image_length / n for some integer n.  For each candidate n in
    [2, image_length // min_period], we score how well a regular grid with
    period image_length/n aligns with the detected breaks (optimising over
    phase).  The best-scoring integer n gives the exact pixel period.

    Returns None if no candidate scores above the minimum confidence (0.3).
    """
    if len(positions) < 2:
        return None
    max_art = int(image_length // min_period)
    if max_art < 2:
        return None

    best_n, best_score = None, 0.0
    for n in range(2, max_art + 1):
        s = _score_art_size(positions, image_length, n, tolerance)
        if s > best_score:
            best_score, best_n = s, n

    if best_n is None or best_score < 0.3:
        return None
    return image_length / best_n


# ---------------------------------------------------------------------------
# Break-list refinement using the smoothness constraint
# ---------------------------------------------------------------------------

def _fill_missing_breaks(
    positions: np.ndarray,
    period: float,
    max_gap_ratio: float = 1.6,
) -> np.ndarray:
    """
    Insert synthetic breaks into gaps that are too wide for a single period.

    Pixel art sizes never vary wildly: a gap of ~2x the period almost certainly
    contains one missed break, a gap of ~3x contains two, etc.  Any gap wider
    than max_gap_ratio * period is subdivided into round(gap / period) equal
    intervals.
    """
    filled = sorted(positions.tolist())
    i = 0
    while i < len(filled) - 1:
        gap = filled[i + 1] - filled[i]
        if gap > period * max_gap_ratio:
            n_insert = round(gap / period) - 1
            if n_insert > 0:
                step = gap / (n_insert + 1)
                new_breaks = [filled[i] + step * (j + 1) for j in range(n_insert)]
                filled = filled[:i + 1] + new_breaks + filled[i + 1:]
                i += n_insert
        i += 1
    return np.array(filled)


def _remove_close_breaks(positions: np.ndarray, period: float, min_ratio: float = 0.5) -> np.ndarray:
    """
    Remove break positions that are too close together (< min_ratio * period).

    Keeps the first of any cluster of closely-spaced breaks (likely double
    detections at the same boundary).
    """
    if len(positions) < 2:
        return positions
    min_spacing = period * min_ratio
    filtered = [positions[0]]
    for p in positions[1:]:
        if p - filtered[-1] >= min_spacing:
            filtered.append(p)
    return np.array(filtered)


# ---------------------------------------------------------------------------
# Grid detection (combines both axes)
# ---------------------------------------------------------------------------

def detect_pixel_grid(
    mag_h: np.ndarray,
    mag_v: np.ndarray,
    threshold_percentile: float,
) -> tuple[float | None, float | None, np.ndarray, np.ndarray]:
    """
    Detect the virtual pixel grid via horizontal and vertical break-line detection.

    Returns
    -------
    pixel_w : estimated virtual pixel width  (None if undetected)
    pixel_h : estimated virtual pixel height (None if undetected)
    col_breaks : column indices of detected vertical break lines  (into W-1 space)
    row_breaks : row indices of detected horizontal break lines   (into H-1 space)
    """
    H = mag_v.shape[0] + 1
    W = mag_h.shape[1] + 1

    # Row breaks — peaks in fraction-of-active-columns voting for a vertical edge at each row
    row_profile = compute_break_profile(mag_v, threshold_percentile, along_axis=1)
    row_breaks  = find_break_positions(row_profile)
    pixel_h     = estimate_pixel_period(row_breaks, H)
    if pixel_h is not None:
        row_breaks = _remove_close_breaks(row_breaks, pixel_h)
        row_breaks = _fill_missing_breaks(row_breaks, pixel_h)

    # Column breaks — peaks in fraction-of-active-rows voting for a horizontal edge at each col
    col_profile = compute_break_profile(mag_h, threshold_percentile, along_axis=0)
    col_breaks  = find_break_positions(col_profile)
    pixel_w     = estimate_pixel_period(col_breaks, W)
    if pixel_w is not None:
        col_breaks = _remove_close_breaks(col_breaks, pixel_w)
        col_breaks = _fill_missing_breaks(col_breaks, pixel_w)

    return pixel_w, pixel_h, col_breaks, row_breaks


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

def count_palette_colors(img: Image.Image, max_colors: int = 256) -> int:
    """
    Quantize the image to at most `max_colors` colours and return how many
    distinct palette entries are actually used.
    """
    quantized = img.quantize(
        colors=max_colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    return len(set(quantized.get_flattened_data()))


# ---------------------------------------------------------------------------
# Output images
# ---------------------------------------------------------------------------

def make_grid_overlay(
    original: Image.Image,
    col_breaks: np.ndarray,
    row_breaks: np.ndarray,
    row_color: tuple[int, int, int] = (255, 50, 50),
    col_color: tuple[int, int, int] = (50, 50, 255),
    alpha: float = 0.85,
) -> Image.Image:
    """
    Draw detected break lines over the original image.

    Horizontal lines (row breaks) are drawn in red; vertical lines in blue.
    Each break at index i is drawn at pixel i+1 (the first row/column of the
    next virtual pixel).
    """
    out = np.array(original, dtype=np.float32)
    cr  = np.array(row_color, dtype=np.float32)
    cc  = np.array(col_color, dtype=np.float32)
    H, W = out.shape[:2]

    for r in row_breaks:
        r1 = int(r) + 1
        if 0 <= r1 < H:
            out[r1, :] = out[r1, :] * (1.0 - alpha) + cr * alpha

    for c in col_breaks:
        c1 = int(c) + 1
        if 0 <= c1 < W:
            out[:, c1] = out[:, c1] * (1.0 - alpha) + cc * alpha

    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def make_grid_only(
    original_size: tuple[int, int],
    col_breaks: np.ndarray,
    row_breaks: np.ndarray,
) -> Image.Image:
    """Return a black image with detected grid lines drawn in white."""
    H, W = original_size
    arr = np.zeros((H, W), dtype=np.uint8)

    for r in row_breaks:
        r1 = int(r) + 1
        if 0 <= r1 < H:
            arr[r1, :] = 255

    for c in col_breaks:
        c1 = int(c) + 1
        if 0 <= c1 < W:
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
    col_breaks: np.ndarray,
    row_breaks: np.ndarray,
    palette_size: int,
) -> None:
    sep = "-" * 46
    print(f"\n{sep}")
    print("  Detection Report")
    print(sep)
    print(f"  Original resolution    : {W} x {H} px")

    # Row breaks
    print(f"  Row breaks detected    : {len(row_breaks)}")
    if len(row_breaks) > 1:
        sp = np.diff(row_breaks.astype(float))
        sp = sp[sp >= 2]
        if len(sp):
            print(f"  Row spacing (mean/std) : {sp.mean():.1f} +/- {sp.std():.1f} px")
    if pixel_h:
        print(f"  Virtual pixel height   : ~{pixel_h:.3f} px")

    # Column breaks
    print(f"  Col breaks detected    : {len(col_breaks)}")
    if len(col_breaks) > 1:
        sp = np.diff(col_breaks.astype(float))
        sp = sp[sp >= 2]
        if len(sp):
            print(f"  Col spacing (mean/std) : {sp.mean():.1f} +/- {sp.std():.1f} px")
    if pixel_w:
        print(f"  Virtual pixel width    : ~{pixel_w:.3f} px")

    # Summary
    print(sep)
    if pixel_w and pixel_h:
        art_w = max(1, round(W / pixel_w))
        art_h = max(1, round(H / pixel_h))
        ds_w  = art_w / W * 100
        ds_h  = art_h / H * 100
        print(f"  Pixel-art resolution   : {art_w} x {art_h} px")
        print(f"  Downsampling ratio     : {ds_w:.2f}% width / {ds_h:.2f}% height")
    else:
        print("  Pixel-art resolution   : unknown")
        print("  Downsampling ratio     : unknown")

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
    edge_percentile = float(args["--edge-percentile"])
    palette_colors  = int(args["--palette-colors"])
    edge_only       = args["--edge-only"]

    print(f"Loading: {input_path}")
    img, arr = load_image(str(input_path))
    H, W = arr.shape[:2]
    print(f"  {W} x {H} px")

    print("Computing colour gradients ...")
    mag_h, mag_v = compute_gradients(arr)

    print("Detecting break lines ...")
    pixel_w, pixel_h, col_breaks, row_breaks = detect_pixel_grid(
        mag_h, mag_v, edge_percentile
    )

    print("Counting colour palette ...")
    palette_size = count_palette_colors(img, palette_colors)

    print_report(W, H, pixel_w, pixel_h, col_breaks, row_breaks, palette_size)

    if edge_only:
        out_img = make_grid_only((H, W), col_breaks, row_breaks)
    else:
        out_img = make_grid_overlay(img, col_breaks, row_breaks)

    print(f"\nSaving: {output_path}")
    out_img.save(str(output_path))
    print("Done.")


if __name__ == "__main__":
    main()
