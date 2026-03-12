#!/usr/bin/env python3
"""Detect the virtual pixel grid using improved multi-threshold consensus.

Enhanced version of detect.py with:
  - Multi-threshold consensus break detection (stable across percentiles)
  - Adaptive gap-filling (works without global period estimate)
  - 2D strip-based break refinement (catches breaks missed by global 1D projection)

Usage:
    detect2.py <input> [--output=PATH] [--edge-percentile=P] [--palette-colors=N] [-g R] [--edge-only]
    detect2.py -h | --help

Arguments:
    <input>                 Input image path.

Options:
    -o PATH --output=PATH       Output image path (default: <input>_edges2.png).
    --edge-percentile=P         Central gradient percentile for consensus range [default: 85].
    --palette-colors=N          Max colours for palette quantisation [default: 256].
    -g R --max-gap=R            Max gap ratio before subdividing [default: 1.6].
    --edge-only                 Output a greyscale grid-line image instead of a colour overlay.
    -h --help                   Show this screen.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
from docopt import docopt
from PIL import Image

from detect import (
    load_image,
    compute_gradients,
    compute_break_profile,
    find_break_positions,
    estimate_pixel_period,
    _remove_close_breaks,
    _fill_missing_breaks,
    count_palette_colors,
    make_grid_overlay,
    make_grid_only,
    print_report,
)


# ---------------------------------------------------------------------------
# Algorithm A: Multi-threshold consensus break detection
# ---------------------------------------------------------------------------

def consensus_break_positions(
    mag: np.ndarray,
    center_percentile: float,
    along_axis: int,
    n_thresholds: int = 7,
    spread: float = 15.0,
    cluster_radius: float = 2.0,
    min_vote_frac: float = 0.4,
) -> np.ndarray:
    """
    Detect break positions via multi-threshold consensus.

    Runs break detection at several percentile thresholds spanning
    [center - spread, center + spread] and keeps positions that appear
    consistently across at least min_vote_frac of the thresholds.

    A genuine virtual-pixel boundary produces a gradient peak at many
    thresholds; noise or anti-aliasing artefacts only fire at the lowest
    thresholds and are filtered out.

    Parameters
    ----------
    mag : ndarray
        Gradient magnitude array (mag_v for row breaks, mag_h for col breaks).
    center_percentile : float
        Central percentile around which thresholds are spread.
    along_axis : int
        Axis for break profile (1 = row breaks from mag_v, 0 = col breaks from mag_h).
    n_thresholds : int
        Number of threshold levels to test.
    spread : float
        Half-width of the percentile range.
    cluster_radius : float
        Maximum distance (in pixels) to merge nearby detections into one cluster.
    min_vote_frac : float
        Minimum fraction of thresholds at which a position must appear.
    """
    lo = max(50.0, center_percentile - spread)
    hi = min(99.0, center_percentile + spread)
    percentiles = np.linspace(lo, hi, n_thresholds)

    # Collect peaks with their originating threshold index
    peak_data: list[tuple[int, int]] = []  # (position, threshold_index)
    for ti, pct in enumerate(percentiles):
        profile = compute_break_profile(mag, pct, along_axis)
        peaks = find_break_positions(profile)
        for p in peaks:
            peak_data.append((int(p), ti))

    if not peak_data:
        return np.array([], dtype=int)

    # Sort by position for greedy clustering
    peak_data.sort()

    # Cluster nearby positions and count distinct thresholds per cluster
    clusters: list[tuple[list[int], set[int]]] = []
    cur_positions = [peak_data[0][0]]
    cur_thresholds = {peak_data[0][1]}

    for pos, ti in peak_data[1:]:
        if pos - cur_positions[-1] <= cluster_radius:
            cur_positions.append(pos)
            cur_thresholds.add(ti)
        else:
            clusters.append((cur_positions, cur_thresholds))
            cur_positions = [pos]
            cur_thresholds = {ti}
    clusters.append((cur_positions, cur_thresholds))

    # Keep clusters with enough distinct-threshold votes
    min_votes = max(2, round(n_thresholds * min_vote_frac))
    result = []
    for positions, thresholds in clusters:
        if len(thresholds) >= min_votes:
            result.append(int(round(np.median(positions))))

    return np.array(sorted(set(result)), dtype=int)


# ---------------------------------------------------------------------------
# Algorithm C: Adaptive gap-filling (works without global period)
# ---------------------------------------------------------------------------

def adaptive_fill_breaks(
    positions: np.ndarray,
    image_length: int,
    period: float | None = None,
    max_gap_ratio: float = 1.6,
) -> np.ndarray:
    """
    Fill gaps in detected breaks using a period estimate.

    If a global period is available, use it directly (same as v1).
    Otherwise, fall back to the median inter-break spacing as an
    approximate period.  This enables gap-filling even when the
    integer art-size sweep fails to converge.
    """
    if len(positions) < 2:
        return positions

    if period is None:
        spacings = np.diff(positions.astype(float))
        period = float(np.median(spacings))
        if period < 3.0:
            return positions

    return _fill_missing_breaks(positions, period, image_length, max_gap_ratio)


# ---------------------------------------------------------------------------
# Algorithm D: 2D strip-based break refinement
# ---------------------------------------------------------------------------

def _breaks_to_spans_local(breaks: np.ndarray, length: int) -> list[tuple[int, int]]:
    """Convert break indices to (start, end) pixel spans."""
    bounds = [0] + [int(b) + 1 for b in sorted(breaks)] + [length]
    return [
        (bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i + 1] > bounds[i]
    ]


def refine_breaks_2d(
    mag_h: np.ndarray,
    mag_v: np.ndarray,
    row_breaks: np.ndarray,
    col_breaks: np.ndarray,
    threshold_percentile: float,
    min_strip_size: int = 6,
    min_strip_votes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine break positions using per-strip detection along the perpendicular axis.

    For column breaks: divide the image into horizontal strips (between row
    breaks), detect column breaks within each strip independently, and add
    positions that appear in multiple strips but were missed by the global
    1D projection.

    Similarly for row breaks using vertical strips (between column breaks).

    This catches breaks that are diluted in the global projection because
    they only span part of the image (e.g. sprite content on a large
    uniform background).
    """
    H = mag_v.shape[0] + 1
    W = mag_h.shape[1] + 1

    # --- Refine column breaks using horizontal strips ---
    row_spans = _breaks_to_spans_local(row_breaks, H)
    col_votes: Counter = Counter()
    n_row_strips = 0

    for r0, r1 in row_spans:
        if r1 - r0 < min_strip_size:
            continue
        n_row_strips += 1
        strip_mag = mag_h[r0:r1, :]
        profile = compute_break_profile(strip_mag, threshold_percentile, along_axis=0)
        peaks = find_break_positions(profile)
        for p in peaks:
            col_votes[int(p)] += 1

    existing_cols = set(int(c) for c in col_breaks)
    new_cols = set(existing_cols)
    eff_min = min(min_strip_votes, max(1, n_row_strips // 3))
    for pos, count in col_votes.items():
        if count >= eff_min and pos not in existing_cols:
            if all(abs(pos - e) > 2 for e in new_cols):
                new_cols.add(pos)

    # --- Refine row breaks using vertical strips ---
    col_spans = _breaks_to_spans_local(col_breaks, W)
    row_votes: Counter = Counter()
    n_col_strips = 0

    for c0, c1 in col_spans:
        if c1 - c0 < min_strip_size:
            continue
        n_col_strips += 1
        strip_mag = mag_v[:, c0:c1]
        profile = compute_break_profile(strip_mag, threshold_percentile, along_axis=1)
        peaks = find_break_positions(profile)
        for p in peaks:
            row_votes[int(p)] += 1

    existing_rows = set(int(r) for r in row_breaks)
    new_rows = set(existing_rows)
    eff_min = min(min_strip_votes, max(1, n_col_strips // 3))
    for pos, count in row_votes.items():
        if count >= eff_min and pos not in existing_rows:
            if all(abs(pos - e) > 2 for e in new_rows):
                new_rows.add(pos)

    return (
        np.array(sorted(new_cols), dtype=float),
        np.array(sorted(new_rows), dtype=float),
    )


# ---------------------------------------------------------------------------
# Combined detection pipeline
# ---------------------------------------------------------------------------

def detect_pixel_grid_v2(
    mag_h: np.ndarray,
    mag_v: np.ndarray,
    threshold_percentile: float,
    max_gap_ratio: float = 1.6,
) -> tuple[float | None, float | None, np.ndarray, np.ndarray]:
    """
    Detect the virtual pixel grid using the improved v2 pipeline.

    Steps:
      1. Multi-threshold consensus break detection
      2. Period estimation (integer art-size sweep, same as v1)
      3. Remove close duplicates (if period known)
      4. Adaptive gap-filling with fallback to median spacing
      5. 2D strip-based refinement
      6. Final duplicate cleanup

    Signature matches detect.detect_pixel_grid so it can be used as a
    drop-in replacement.

    Returns
    -------
    pixel_w, pixel_h : estimated virtual pixel dimensions (None if undetected)
    col_breaks, row_breaks : detected break positions
    """
    H = mag_v.shape[0] + 1
    W = mag_h.shape[1] + 1

    # Step 1: Consensus break detection
    row_breaks = consensus_break_positions(mag_v, threshold_percentile, along_axis=1)
    col_breaks = consensus_break_positions(mag_h, threshold_percentile, along_axis=0)

    # Step 2: Period estimation
    pixel_h = estimate_pixel_period(row_breaks, H)
    pixel_w = estimate_pixel_period(col_breaks, W)

    # Step 3: Remove close duplicates
    if pixel_h is not None:
        row_breaks = _remove_close_breaks(row_breaks, pixel_h)
    if pixel_w is not None:
        col_breaks = _remove_close_breaks(col_breaks, pixel_w)

    # Step 4: Adaptive gap-filling (works even without a global period)
    row_breaks = adaptive_fill_breaks(row_breaks, H, pixel_h, max_gap_ratio)
    col_breaks = adaptive_fill_breaks(col_breaks, W, pixel_w, max_gap_ratio)

    # Step 5: 2D strip-based refinement
    col_breaks, row_breaks = refine_breaks_2d(
        mag_h, mag_v, row_breaks, col_breaks, threshold_percentile,
    )

    # Step 6: Final cleanup
    if pixel_h is not None:
        row_breaks = _remove_close_breaks(row_breaks, pixel_h)
    if pixel_w is not None:
        col_breaks = _remove_close_breaks(col_breaks, pixel_w)

    return pixel_w, pixel_h, col_breaks, row_breaks


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
        input_path.parent / (input_path.stem + "_edges2.png")
    )
    edge_percentile = float(args["--edge-percentile"])
    palette_colors  = int(args["--palette-colors"])
    max_gap_ratio   = float(args["--max-gap"])
    edge_only       = args["--edge-only"]

    print(f"Loading: {input_path}")
    img, arr = load_image(str(input_path))
    H, W = arr.shape[:2]
    print(f"  {W} x {H} px")

    print("Computing colour gradients ...")
    mag_h, mag_v = compute_gradients(arr)

    print("Detecting break lines (v2 consensus) ...")
    pixel_w, pixel_h, col_breaks, row_breaks = detect_pixel_grid_v2(
        mag_h, mag_v, edge_percentile, max_gap_ratio,
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
