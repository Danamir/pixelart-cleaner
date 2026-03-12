#!/usr/bin/env python3
"""Downsample AI-generated pixel art to true pixel-art resolution.

Uses the v3 luma line-pair detection algorithm to find virtual pixel
boundaries, then samples each virtual pixel block using the chosen method.

Usage:
    resize3.py <input>... [options]
    resize3.py -h | --help

Arguments:
    <input>...              Input image path(s).

Options:
    -o PATH --output=PATH       Output path (default: <stem>_pixel3.png).
    -p P --edge-percentile=P    Per-pixel luma-diff percentile threshold [default: 85].
    -e E --min-edge=E           Minimum absolute luma diff to count as an edge [default: 10].
    -d D --min-distance=D       Minimum distance in px between two peaks in the break profile [default: 3].
    -c C --cluster-radius=C     Max distance in px to merge nearby band votes into one break [default: 1].
    -g R --max-gap=R            Max gap ratio before subdividing [default: 1.6].
    --regular-tolerance=T       Span-size tolerance for regularity check (0-1) [default: 0.10].
    -r --force-regular          Always use regular grid strategy.
    -i --force-irregular        Always use irregular span strategy.
    -s S --scale=S              Divide detected pixel size by S (S>1 finer, S<1 coarser) [default: 1.0].
    -t SIZE --tile=SIZE         Process in independent tiles, e.g. 64x64.
    -m METHOD --sample=METHOD   Color sampling method: center, center_region, max [default: center].
    -q --square                 Force square output pixels.
    -v --verbose                Also save _edges3, _compare3, _compare3_true images.
    -h --help                   Show this screen.
"""

import sys
from math import ceil
from pathlib import Path

import numpy as np
from docopt import docopt
from PIL import Image

from detect3 import (
    load_image,
    detect_pixel_grid_v3,
    make_grid_overlay,
)


# ---------------------------------------------------------------------------
# Block sampling
# ---------------------------------------------------------------------------

def _sample_center(block: np.ndarray) -> np.ndarray:
    """Return the single center pixel of an RGB block (H, W, 3)."""
    h, w = block.shape[:2]
    return block[h // 2, w // 2].astype(np.float32)


def _sample_max(block: np.ndarray, n_buckets: int = 10) -> np.ndarray:
    """
    Return the mean color of the dominant color cluster in the block.

    Quantizes each RGB channel into n_buckets levels, finds the bucket
    (channel triplet) with the most pixels, then returns the mean of the
    actual (unquantized) pixels belonging to that bucket.  This is robust
    to the slight per-pixel color variation typical in AI-generated pixel art.
    """
    pixels = block.reshape(-1, 3)
    step = max(1, round(256 / n_buckets))
    quantized = (pixels // step).astype(np.int32)
    keys, counts = np.unique(quantized, axis=0, return_counts=True)
    best = keys[counts.argmax()]
    mask = np.all(quantized == best, axis=1)
    return pixels[mask].mean(axis=0)


def _sample_center_region(block: np.ndarray) -> np.ndarray:
    """
    Return the mean color of the inner 50% of an RGB block (H, W, 3).

    Trims 25% from each edge to avoid anti-aliased or blended boundary pixels.
    Falls back to the full block for blocks too small to trim meaningfully.
    """
    h, w = block.shape[:2]
    r0, r1 = h // 4, h - h // 4
    c0, c1 = w // 4, w - w // 4
    if r0 >= r1:
        r0, r1 = 0, h
    if c0 >= c1:
        c0, c1 = 0, w
    return block[r0:r1, c0:c1].mean(axis=(0, 1))


def sample_block(block: np.ndarray, method: str = "center") -> np.ndarray:
    """Dispatch to the chosen sampling method."""
    if method == "center_region":
        return _sample_center_region(block)
    if method == "max":
        return _sample_max(block)
    return _sample_center(block)


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

def spans_from_breaks(breaks: np.ndarray, image_length: int) -> list[tuple[int, int]]:
    """Convert break positions (gradient space) to (start, end) pixel spans."""
    bounds = [0] + [int(b) + 1 for b in sorted(breaks)] + [image_length]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]


def _best_phase(boundaries: np.ndarray, period: float) -> float:
    """
    Find the phase of a regular grid (position of first boundary) that best
    aligns to the observed span boundaries using a circular mean.
    """
    if len(boundaries) == 0:
        return 0.0
    mapped = boundaries % period
    angles = mapped / period * 2 * np.pi
    cx = float(np.cos(angles).mean())
    cy = float(np.sin(angles).mean())
    angle = np.arctan2(cy, cx)
    if angle < 0:
        angle += 2 * np.pi
    return float(angle / (2 * np.pi) * period)


def regular_grid_spans(
    image_length: int,
    period: float,
    breaks: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Generate regular-grid (start, end) spans covering [0, image_length),
    aligned to the detected break positions via circular-mean phase fitting.
    """
    if len(breaks) > 0:
        boundaries = breaks.astype(float) + 1   # pixel-space boundaries
        phase = _best_phase(boundaries, period)
    else:
        phase = 0.0

    # Find the first span index k such that phase + k*period <= 0
    if phase > 0:
        k_first = -int(ceil(phase / period))
    else:
        k_first = 0

    spans = []
    k = k_first
    while True:
        pos_s = phase + k * period
        pos_e = phase + (k + 1) * period
        if pos_s >= image_length:
            break
        s = max(0, round(pos_s))
        e = min(image_length, round(pos_e))
        if e > s:
            spans.append((s, e))
        k += 1

    return spans


# ---------------------------------------------------------------------------
# Core downsampling
# ---------------------------------------------------------------------------

def downsample(
    arr: np.ndarray,
    col_spans: list[tuple[int, int]],
    row_spans: list[tuple[int, int]],
    method: str = "center",
) -> np.ndarray:
    """
    Sample each virtual pixel block (row_span x col_span).

    Returns float32 array of shape (len(row_spans), len(col_spans), 3).
    """
    out = np.zeros((len(row_spans), len(col_spans), 3), dtype=np.float32)
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            out[ri, ci] = sample_block(arr[r0:r1, c0:c1], method)
    return out


# ---------------------------------------------------------------------------
# Square-pixel irregular mode (repeats output pixels for wide/tall spans)
# ---------------------------------------------------------------------------

def downsample_square_irregular(
    arr: np.ndarray,
    col_spans: list[tuple[int, int]],
    row_spans: list[tuple[int, int]],
    target_size: float,
    method: str = "center",
) -> np.ndarray:
    """
    Irregular downsampling with square-pixel output.

    Each span is sampled once; its color is repeated round(span/target_size)
    times so that every output pixel represents an approximately square region.
    """
    col_colors = []
    for c0, c1 in col_spans:
        reps = max(1, round((c1 - c0) / target_size))
        col_colors.append((c0, c1, reps))

    row_colors = []
    for r0, r1 in row_spans:
        reps = max(1, round((r1 - r0) / target_size))
        row_colors.append((r0, r1, reps))

    total_rows = sum(r for _, _, r in row_colors)
    total_cols = sum(r for _, _, r in col_colors)
    out = np.zeros((total_rows, total_cols, 3), dtype=np.float32)

    row_out = 0
    for r0, r1, rreps in row_colors:
        col_out = 0
        for c0, c1, creps in col_colors:
            color = sample_block(arr[r0:r1, c0:c1], method)
            for dr in range(rreps):
                for dc in range(creps):
                    out[row_out + dr, col_out + dc] = color
            col_out += creps
        row_out += rreps

    return out


# ---------------------------------------------------------------------------
# Single-image downsampling
# ---------------------------------------------------------------------------

def _downsample_single(
    arr: np.ndarray,
    threshold_percentile: float,
    max_gap_ratio: float,
    regular_tolerance: float,
    force_regular: bool,
    force_irregular: bool,
    scale: float,
    square: bool,
    min_luma_diff: float = 10.0,
    method: str = "center",
    min_distance: int = 3,
    cluster_radius: int = 1,
) -> np.ndarray:
    """Detect grid and downsample one image (or tile)."""
    pixel_w, pixel_h, is_regular_w, is_regular_h, col_breaks, row_breaks, _, _ = \
        detect_pixel_grid_v3(arr, threshold_percentile, max_gap_ratio, regular_tolerance, min_luma_diff,
                             min_distance=min_distance, cluster_radius=cluster_radius)

    # Apply scale
    if pixel_w is not None:
        pixel_w /= scale
    if pixel_h is not None:
        pixel_h /= scale

    # Decide strategy
    both_regular = is_regular_w and is_regular_h
    use_regular = (
        not force_irregular
        and (force_regular or both_regular)
        and pixel_w is not None
        and pixel_h is not None
    )

    H, W = arr.shape[:2]

    if use_regular:
        if square:
            sq = (pixel_w + pixel_h) / 2.0
            pixel_w = pixel_h = sq
        col_spans = regular_grid_spans(W, pixel_w, col_breaks)
        row_spans = regular_grid_spans(H, pixel_h, row_breaks)
        return downsample(arr, col_spans, row_spans, method)
    else:
        col_spans = spans_from_breaks(col_breaks, W)
        row_spans = spans_from_breaks(row_breaks, H)
        if square and pixel_w is not None and pixel_h is not None:
            target = min(pixel_w, pixel_h) / scale
            return downsample_square_irregular(arr, col_spans, row_spans, target, method)
        return downsample(arr, col_spans, row_spans, method)


# ---------------------------------------------------------------------------
# Tiled downsampling
# ---------------------------------------------------------------------------

def _parse_tile_size(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    return int(parts[0]), int(parts[1])


def _downsample_tiled(
    arr: np.ndarray,
    tile_w: int,
    tile_h: int,
    threshold_percentile: float,
    max_gap_ratio: float,
    regular_tolerance: float,
    force_regular: bool,
    force_irregular: bool,
    scale: float,
    square: bool,
    min_luma_diff: float = 10.0,
    method: str = "center",
    min_distance: int = 3,
    cluster_radius: int = 1,
) -> np.ndarray:
    """
    Split the image into tiles, downsample each independently, then assemble.

    Each tile gets its own grid detection, providing local phase correction.
    Tiles in the same row are padded to the same output height; tiles in the
    same column are padded to the same output width before concatenation.
    """
    H, W = arr.shape[:2]
    row_bounds = [(r, min(r + tile_h, H)) for r in range(0, H, tile_h)]
    col_bounds = [(c, min(c + tile_w, W)) for c in range(0, W, tile_w)]

    out_rows = []
    for r0, r1 in row_bounds:
        out_cols = []
        for c0, c1 in col_bounds:
            tile = arr[r0:r1, c0:c1]
            tile_out = _downsample_single(
                tile, threshold_percentile, max_gap_ratio, regular_tolerance,
                force_regular, force_irregular, scale, square, min_luma_diff, method,
                min_distance, cluster_radius,
            )
            out_cols.append(tile_out)

        # Pad all tiles in this row-strip to the same output height
        max_h = max(t.shape[0] for t in out_cols)
        padded = []
        for t in out_cols:
            if t.shape[0] < max_h:
                pad = np.zeros((max_h - t.shape[0], t.shape[1], 3), dtype=np.float32)
                t = np.concatenate([t, pad], axis=0)
            padded.append(t)
        out_rows.append(np.concatenate(padded, axis=1))

    # Pad all row-strips to the same output width
    max_w = max(r.shape[1] for r in out_rows)
    padded_rows = []
    for r in out_rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.float32)
            r = np.concatenate([r, pad], axis=1)
        padded_rows.append(r)

    return np.concatenate(padded_rows, axis=0)


# ---------------------------------------------------------------------------
# Verbose output helpers
# ---------------------------------------------------------------------------

def _build_output_stem(stem: str, args: dict) -> str:
    """Append active non-default option suffixes to the output stem."""
    suffix = ""
    if args["--force-irregular"]:
        suffix += "_i"
    if args["--force-regular"]:
        suffix += "_r"
    if args["--square"]:
        suffix += "_q"
    g = float(args["--max-gap"])
    if g != 1.6:
        suffix += f"_g{g:g}"
    p = float(args["--edge-percentile"])
    if p != 85:
        suffix += f"_p{p:g}"
    return stem + suffix


def _save_verbose(
    img_original: Image.Image,
    out_arr: np.ndarray,
    col_breaks: np.ndarray,
    row_breaks: np.ndarray,
    col_synth: np.ndarray,
    row_synth: np.ndarray,
    stem: str,
    output_dir: Path,
) -> None:
    """Save the edges overlay, compare, and compare_true verbose images."""
    H_orig, W_orig = np.array(img_original).shape[:2]

    # Edges overlay
    edges_path = output_dir / (stem + "_edges3.png")
    make_grid_overlay(img_original, col_breaks, row_breaks, col_synth, row_synth).save(str(edges_path))
    print(f"  Saved: {edges_path}")

    # Compare: output scaled back to original size
    out_pil = Image.fromarray(out_arr.clip(0, 255).astype(np.uint8))
    compare = out_pil.resize((W_orig, H_orig), Image.NEAREST)
    compare_path = output_dir / (stem + "_compare3.png")
    compare.save(str(compare_path))
    print(f"  Saved: {compare_path}")

    # Compare true: output scaled up by largest integer factor that fits
    out_h, out_w = out_arr.shape[:2]
    if out_w > 0 and out_h > 0:
        factor = max(1, min(W_orig // out_w, H_orig // out_h))
        compare_true = out_pil.resize((out_w * factor, out_h * factor), Image.NEAREST)
        ct_path = output_dir / (stem + "_compare3_true.png")
        compare_true.save(str(ct_path))
        print(f"  Saved: {ct_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = docopt(__doc__)

    threshold_percentile = float(args["--edge-percentile"])
    min_luma_diff = float(args["--min-edge"])
    min_distance = int(args["--min-distance"])
    cluster_radius = int(args["--cluster-radius"])
    max_gap_ratio = float(args["--max-gap"])
    regular_tolerance = float(args["--regular-tolerance"])
    method = args["--sample"]
    force_regular = args["--force-regular"]
    force_irregular = args["--force-irregular"]
    scale = float(args["--scale"])
    square = args["--square"]
    verbose = args["--verbose"]
    tile_size = args["--tile"]

    valid_methods = {"center", "center_region", "max"}
    if method not in valid_methods:
        print(f"Error: --sample must be one of: {', '.join(sorted(valid_methods))}", file=sys.stderr)
        sys.exit(1)

    for input_str in args["<input>"]:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"Error: file not found: {input_path}", file=sys.stderr)
            continue

        print(f"\nProcessing: {input_path}")
        img, arr = load_image(str(input_path))
        H, W = arr.shape[:2]
        print(f"  {W} x {H} px")

        # Determine output path
        if args["--output"]:
            output_path = Path(args["--output"])
        else:
            vstem = _build_output_stem(input_path.stem, args)
            output_path = input_path.parent / (vstem + "_pixel3.png")

        # Downsample
        if tile_size:
            tw, th = _parse_tile_size(tile_size)
            print(f"  Tiled mode: {tw}x{th} tiles")
            out_arr = _downsample_tiled(
                arr, tw, th, threshold_percentile, max_gap_ratio,
                regular_tolerance, force_regular, force_irregular, scale, square,
                min_luma_diff, method, min_distance, cluster_radius,
            )
            # For verbose edges overlay, run detection on full image
            pixel_w, pixel_h, is_regular_w, is_regular_h, col_breaks, row_breaks, col_synth, row_synth = \
                detect_pixel_grid_v3(arr, threshold_percentile, max_gap_ratio, regular_tolerance, min_luma_diff,
                                     min_distance=min_distance, cluster_radius=cluster_radius)
        else:
            pixel_w, pixel_h, is_regular_w, is_regular_h, col_breaks, row_breaks, col_synth, row_synth = \
                detect_pixel_grid_v3(arr, threshold_percentile, max_gap_ratio, regular_tolerance, min_luma_diff,
                                     min_distance=min_distance, cluster_radius=cluster_radius)

            mode_str = "regular" if (is_regular_w and is_regular_h and not force_irregular) else "irregular"
            print(f"  Mode: {mode_str}")
            if pixel_w:
                print(f"  Virtual pixel size: ~{pixel_w:.2f} x {pixel_h:.2f} px")

            out_arr = _downsample_single(
                arr, threshold_percentile, max_gap_ratio, regular_tolerance,
                force_regular, force_irregular, scale, square, min_luma_diff, method,
                min_distance, cluster_radius,
            )

        out_h, out_w = out_arr.shape[:2]
        print(f"  Output: {out_w} x {out_h} px")

        # Save main output
        out_pil = Image.fromarray(out_arr.clip(0, 255).astype(np.uint8))
        out_pil.save(str(output_path))
        print(f"  Saved: {output_path}")

        # Verbose outputs
        if verbose:
            vstem = _build_output_stem(input_path.stem, args)
            _save_verbose(img, out_arr, col_breaks, row_breaks, col_synth, row_synth,
                          vstem, input_path.parent)


if __name__ == "__main__":
    main()
