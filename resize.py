#!/usr/bin/env python3
"""Downsample an AI-generated pixel-art image to its true pixel-art resolution.

Detects the virtual pixel grid (via break-line analysis) then reconstructs the
image by sampling each virtual pixel block.  Two strategies are auto-selected:

  regular grid   -- period estimated successfully: nearest-neighbour sampling
                    on a phase-corrected regular grid centred over each virtual
                    pixel.
  irregular grid -- period estimation failed: each virtual pixel is individually
                    bounded by the detected break lines and sampled at its
                    centre (or mean/median).

In addition to the downsampled pixel-art output, two comparison images are saved:
  _compare.png      -- pixel art scaled back to the original image dimensions
                       via nearest-neighbour (exact same pixel count as input).
  _compare_true.png -- pixel art scaled up by the largest integer factor that
                       keeps it close to the original size (preserves square
                       pixel-art pixels).

Usage:
    resize.py <input>... [--output=PATH] [--edge-percentile=P] [--sample=METHOD]
              [-r | -i] [-s S] [-t SIZE] [-q] [-v]
    resize.py -h | --help

Arguments:
    <input>                 Input image path(s).

Options:
    -o PATH --output=PATH       Output image path (default: <input>_pixel.<ext>).
    --edge-percentile=P         Gradient percentile threshold for break detection [default: 85].
    --sample=METHOD             Sampling method for irregular grids: center, mean, median [default: center].
    -r --force-regular          Force regular-grid strategy regardless of detection quality.
    -i --force-irregular        Force irregular span-based strategy regardless of detection quality.
    -s S --scale=S              Divide detected virtual pixel size by S before resampling [default: 1.0].
    -t SIZE --tile=SIZE         Split image into tiles of SIZE (e.g. 64x64) and detect each independently.
                                The global period is kept; only the phase is corrected per tile.
    -q --square                 Force square virtual pixels by averaging detected width and height.
    -v --verbose                Also save _edges.png, _compare.png, and _compare_true.png.
    -h --help                   Show this screen.
"""

import sys
from pathlib import Path

import numpy as np
from docopt import docopt
from PIL import Image

from detect import load_image, compute_gradients, detect_pixel_grid, make_grid_overlay


# ---------------------------------------------------------------------------
# Span construction
# ---------------------------------------------------------------------------

def _spans_in_tile(
    global_spans: list[tuple[int, int]],
    t0: int,
    t1: int,
) -> list[tuple[int, int]]:
    """
    Return the subset of global spans whose CENTER falls in [t0, t1), clipped
    to [t0, t1) and shifted to tile-local coordinates.

    Attributing spans by center guarantees each span belongs to exactly one
    tile even when it straddles a tile boundary, so span counts sum correctly
    across tiles.
    """
    result = []
    for s, e in global_spans:
        if t0 <= (s + e) / 2 < t1:
            local_s = max(s, t0) - t0
            local_e = min(e, t1) - t0
            if local_e > local_s:
                result.append((local_s, local_e))
    return result


def _breaks_to_spans(breaks: np.ndarray, image_length: int) -> list[tuple[int, int]]:
    """
    Convert detected break line indices to (start, end) pixel spans.

    A break at index b means a boundary between pixel b and b+1, so the
    boundary sits at b+1.  Spans cover the full [0, image_length) range.
    """
    bounds = [0] + [int(b) + 1 for b in sorted(breaks)] + [image_length]
    return [
        (bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i + 1] > bounds[i]
    ]


def _spacing_cv(breaks: np.ndarray) -> float:
    """
    Coefficient of variation of spacings between consecutive detected breaks.

    Measures true grid regularity: 0 = perfectly even, high = irregular spacing.
    Ignores total break count (so sparse-but-regular detection stays near 0).
    """
    if len(breaks) < 2:
        return 0.0
    spacings = np.diff(breaks.astype(float))
    return float(spacings.std() / spacings.mean()) if spacings.mean() > 0 else 0.0


# ---------------------------------------------------------------------------
# Regular-grid centers (consistent case)
# ---------------------------------------------------------------------------

def _regular_centers(
    pixel_period: float,
    image_length: int,
    breaks: np.ndarray,
) -> list[int]:
    """
    Compute center sample coordinates for a phase-corrected regular virtual-pixel grid.

    Uses detected break positions to find the best-fitting phase offset (via
    Gaussian alignment), then places one center per virtual pixel across the
    image at half-period offsets from each grid boundary.

    Returns a list of integer pixel coordinates, one per virtual pixel.
    """
    art_size = max(1, round(image_length / pixel_period))

    # --- Phase estimation ---
    if len(breaks) >= 1:
        best_phase, best_score = 0.0, -1.0
        for anchor in breaks:
            phase = float(anchor) % pixel_period
            rem   = (breaks.astype(float) - phase) % pixel_period
            aligned = np.minimum(rem, pixel_period - rem)
            score = float(np.exp(-0.5 * (aligned / 1.0) ** 2).mean())
            if score > best_score:
                best_score, best_phase = score, phase
    else:
        best_phase = 0.0

    # --- Build grid boundaries within [0, image_length] ---
    # Grid lines at: ..., best_phase - p, best_phase, best_phase + p, ...
    # A phase offset > 0 produces a leading partial span [0, best_phase) AND a
    # trailing partial span at the other end, giving art_size + 1 intervals total.
    # We clip back to exactly art_size by dropping the smaller of the two edge spans.
    first = best_phase if best_phase > 0 else pixel_period
    bounds = [0.0]
    b = first
    while b < image_length:
        bounds.append(b)
        b += pixel_period
    bounds.append(float(image_length))

    # Drop the smaller edge span until we reach art_size intervals
    while len(bounds) - 1 > art_size:
        lead  = bounds[1]  - bounds[0]
        trail = bounds[-1] - bounds[-2]
        if lead <= trail:
            bounds.pop(0)
        else:
            bounds.pop(-1)

    # --- One center per virtual pixel span ---
    return [
        max(0, min(image_length - 1, round((bounds[i] + bounds[i + 1]) / 2.0)))
        for i in range(len(bounds) - 1)
    ]


# ---------------------------------------------------------------------------
# Span subdivision (for --scale with irregular mode)
# ---------------------------------------------------------------------------

def _subdivide_spans(
    spans: list[tuple[int, int]],
    target_size: float,
) -> list[tuple[int, int]]:
    """
    Split each span into round(size / target_size) equal sub-spans.
    Used when scale > 1.0 (target_size < natural span size).
    """
    result = []
    for s, e in spans:
        n = max(1, round((e - s) / target_size))
        step = (e - s) / n
        for i in range(n):
            sub_s = round(s + i * step)
            sub_e = round(s + (i + 1) * step)
            if sub_e > sub_s:
                result.append((sub_s, sub_e))
    return result


def _merge_spans(
    spans: list[tuple[int, int]],
    group_size: int,
) -> list[tuple[int, int]]:
    """
    Merge consecutive spans in batches of group_size into a single span.
    Used when scale < 1.0 (group_size = round(1 / scale) virtual pixels
    should map to one output pixel).
    """
    result = []
    for i in range(0, len(spans), group_size):
        batch = spans[i : i + group_size]
        result.append((batch[0][0], batch[-1][1]))
    return result


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_block(
    arr: np.ndarray,
    r0: int, r1: int,
    c0: int, c1: int,
    method: str,
) -> np.ndarray:
    """Return the sampled colour (uint8, shape (3,)) for one virtual pixel block."""
    if method == "center":
        r = (r0 + r1) // 2
        c = (c0 + c1) // 2
        return arr[r, c].clip(0, 255).astype(np.uint8)
    block = arr[r0:r1, c0:c1].reshape(-1, 3)
    if method == "mean":
        return block.mean(axis=0).clip(0, 255).astype(np.uint8)
    # median
    return np.median(block, axis=0).clip(0, 255).astype(np.uint8)


def _downsample_irregular(
    arr: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    method: str,
) -> Image.Image:
    """Reconstruct by sampling each detected virtual pixel block individually."""
    H_out, W_out = len(row_spans), len(col_spans)
    out = np.zeros((H_out, W_out, 3), dtype=np.uint8)
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            out[ri, ci] = _sample_block(arr, r0, r1, c0, c1, method)
    return Image.fromarray(out)


def _downsample_regular(
    arr: np.ndarray,
    row_centers: list[int],
    col_centers: list[int],
) -> Image.Image:
    """Reconstruct by nearest-neighbour sampling at computed grid centers."""
    H_out, W_out = len(row_centers), len(col_centers)
    out = np.zeros((H_out, W_out, 3), dtype=np.uint8)
    for ri, r in enumerate(row_centers):
        for ci, c in enumerate(col_centers):
            out[ri, ci] = arr[r, c].clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


# ---------------------------------------------------------------------------
# Tiled downsampling
# ---------------------------------------------------------------------------

def _parse_tile_size(s: str) -> tuple[int, int]:
    """Parse a 'WxH' tile size string into (width, height) integers."""
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise ValueError(f"Invalid tile size {s!r} — expected format: WxH (e.g. 64x64)")


def _pad_tile(tile: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Pad a tile array to (target_h, target_w) by repeating edge pixels.

    Also trims to the target if the tile is somehow larger (shouldn't happen
    in normal use but guards against rounding edge cases).
    """
    h, w = tile.shape[:2]
    if h < target_h:
        pad = np.repeat(tile[-1:, :], target_h - h, axis=0)
        tile = np.concatenate([tile, pad], axis=0)
    if w < target_w:
        pad = np.repeat(tile[:, -1:], target_w - w, axis=1)
        tile = np.concatenate([tile, pad], axis=1)
    return tile[:target_h, :target_w]


def _downsample_tiled(
    arr: np.ndarray,
    pixel_w: float,
    pixel_h: float,
    col_breaks: np.ndarray,
    row_breaks: np.ndarray,
    tile_w: int,
    tile_h: int,
    edge_percentile: float,
    sample_method: str,
    use_regular: bool,
    scale: float = 1.0,
    square: bool = False,
) -> Image.Image:
    """
    Downsample by processing the image in tiles.

    Two-pass algorithm:
      Pass 1 -- sample every tile independently (local phase correction for
                regular mode; local span detection for irregular mode).
      Pass 2 -- determine the canonical output size per strip, pad tiles that
                came out short, and concatenate.

    Canonical size per strip:
      regular   -- count of global output centers falling in that strip's input
                   range (guarantees exact art_w x art_h total).
      irregular -- maximum output size across all tiles in the strip (no global
                   reference exists; padding fills tiles where detection missed
                   a span).
    """
    H, W = arr.shape[:2]

    tile_row_starts = list(range(0, H, tile_h))
    tile_col_starts = list(range(0, W, tile_w))

    # Pre-compute global spans for irregular mode (used in pass 1)
    global_row_spans = _breaks_to_spans(row_breaks, H) if not use_regular else None
    global_col_spans = _breaks_to_spans(col_breaks, W) if not use_regular else None

    # --- Pass 1: sample every tile ---
    tile_grid: list[list[np.ndarray]] = []
    for r0 in tile_row_starts:
        r1 = min(r0 + tile_h, H)
        th = r1 - r0
        row: list[np.ndarray] = []
        for c0 in tile_col_starts:
            c1 = min(c0 + tile_w, W)
            tw = c1 - c0
            tile_arr = arr[r0:r1, c0:c1]

            if use_regular:
                # Regular mode: local detection for phase correction;
                # fall back to global breaks if the tile is featureless.
                mag_h_t, mag_v_t = compute_gradients(tile_arr)
                _, _, t_col_breaks, t_row_breaks = detect_pixel_grid(
                    mag_h_t, mag_v_t, edge_percentile
                )
                if len(t_row_breaks) == 0:
                    t_row_breaks = row_breaks[(row_breaks >= r0) & (row_breaks < r1)] - r0
                if len(t_col_breaks) == 0:
                    t_col_breaks = col_breaks[(col_breaks >= c0) & (col_breaks < c1)] - c0
                row_centers = _regular_centers(pixel_h, th, t_row_breaks)
                col_centers = _regular_centers(pixel_w, tw, t_col_breaks)
                tile_out = np.array(_downsample_regular(tile_arr, row_centers, col_centers))
            else:
                # Irregular mode: attribute each global span to the tile containing
                # its center to avoid double-counting at tile boundaries.
                t_row_spans = _spans_in_tile(global_row_spans, r0, r1)
                t_col_spans = _spans_in_tile(global_col_spans, c0, c1)
                if scale != 1.0 or square:
                    if scale >= 1.0:
                        t_row_spans = _subdivide_spans(t_row_spans, pixel_h)
                        t_col_spans = _subdivide_spans(t_col_spans, pixel_w)
                    else:
                        g = max(1, round(1.0 / scale))
                        t_row_spans = _merge_spans(t_row_spans, g)
                        t_col_spans = _merge_spans(t_col_spans, g)
                tile_out = np.array(
                    _downsample_irregular(tile_arr, t_row_spans, t_col_spans, sample_method)
                )

            row.append(tile_out)
        tile_grid.append(row)

    n_tile_rows = len(tile_row_starts)
    n_tile_cols = len(tile_col_starts)

    # --- Pass 2: canonical sizes, pad, assemble ---
    if use_regular:
        # Use global grid to guarantee exact art_h x art_w total
        global_row_centers = _regular_centers(pixel_h, H, row_breaks)
        global_col_centers = _regular_centers(pixel_w, W, col_breaks)
        canon_rows = [
            sum(1 for rc in global_row_centers if r0 <= rc < min(r0 + tile_h, H))
            for r0 in tile_row_starts
        ]
        canon_cols = [
            sum(1 for cc in global_col_centers if c0 <= cc < min(c0 + tile_w, W))
            for c0 in tile_col_starts
        ]
    else:
        # No global reference: use the maximum output size within each strip
        canon_rows = [
            max(tile_grid[ri][ci].shape[0] for ci in range(n_tile_cols))
            for ri in range(n_tile_rows)
        ]
        canon_cols = [
            max(tile_grid[ri][ci].shape[1] for ri in range(n_tile_rows))
            for ci in range(n_tile_cols)
        ]

    strip_list = []
    for ri in range(n_tile_rows):
        if canon_rows[ri] == 0:
            continue
        tile_row_out = []
        for ci in range(n_tile_cols):
            if canon_cols[ci] == 0:
                continue
            tile_out = _pad_tile(tile_grid[ri][ci], canon_rows[ri], canon_cols[ci])
            tile_row_out.append(tile_out)
        if tile_row_out:
            strip_list.append(np.concatenate(tile_row_out, axis=1))

    return Image.fromarray(np.concatenate(strip_list, axis=0))


# ---------------------------------------------------------------------------
# Comparison images
# ---------------------------------------------------------------------------

def make_compare(pixel_img: Image.Image, orig_W: int, orig_H: int) -> Image.Image:
    """Scale pixel art back to the original image dimensions (nearest-neighbour)."""
    return pixel_img.resize((orig_W, orig_H), Image.Resampling.NEAREST)


def make_compare_true(pixel_img: Image.Image, orig_W: int, orig_H: int) -> Image.Image:
    """
    Scale pixel art by the nearest integer factor relative to the original size.

    The scale is chosen as round(min(orig_W / art_W, orig_H / art_H)), so each
    pixel-art pixel maps to exactly scale x scale output pixels (square pixels).
    """
    art_W, art_H = pixel_img.size
    scale = max(1, round(min(orig_W / art_W, orig_H / art_H)))
    return pixel_img.resize((art_W * scale, art_H * scale), Image.Resampling.NEAREST)


# ---------------------------------------------------------------------------
# Verbose filename suffix
# ---------------------------------------------------------------------------

def _flags_suffix(
    force_regular: bool,
    force_irregular: bool,
    scale: float,
    tile_size: tuple[int, int] | None,
    square: bool = False,
) -> str:
    """
    Build a filename suffix encoding the active non-default options, e.g.
    '_r_s2.0_t16x16'.  Returns an empty string when all options are at
    their defaults.
    """
    parts = []
    if force_regular:
        parts.append("r")
    if force_irregular:
        parts.append("i")
    if square:
        parts.append("q")
    if scale != 1.0:
        parts.append(f"s{scale}")
    if tile_size is not None:
        parts.append(f"t{tile_size[0]}x{tile_size[1]}")
    return ("_" + "_".join(parts)) if parts else ""


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(
    W: int, H: int,
    W_out: int, H_out: int,
    row_cv: float, col_cv: float,
    strategy: str,
    compare_true_scale: int,
) -> None:
    sep = "-" * 46
    print(f"\n{sep}")
    print("  Resize Report")
    print(sep)
    print(f"  Input  resolution : {W} x {H} px")
    print(f"  Output resolution : {W_out} x {H_out} px")
    print(f"  Downsampling      : {W_out/W*100:.2f}% width / {H_out/H*100:.2f}% height")
    row_tag = "consistent" if row_cv < 0.25 else "irregular"
    col_tag = "consistent" if col_cv < 0.25 else "irregular"
    print(f"  Row spacing CV    : {row_cv:.3f}  ({row_tag})")
    print(f"  Col spacing CV    : {col_cv:.3f}  ({col_tag})")
    print(f"  Strategy          : {strategy}")
    print(f"  Compare true scale: {compare_true_scale}x  "
          f"({W_out * compare_true_scale} x {H_out * compare_true_scale} px)")
    print(sep)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    input_path: Path,
    output_path: Path,
    edge_percentile: float,
    sample_method: str,
    force_regular: bool,
    force_irregular: bool,
    scale: float,
    verbose: bool,
    tile_size: tuple[int, int] | None = None,
    flags_suffix: str = "",
    square: bool = False,
) -> None:
    print(f"Loading: {input_path}")
    img, arr = load_image(str(input_path))
    H, W = arr.shape[:2]
    print(f"  {W} x {H} px")

    print("Detecting virtual pixel grid ...")
    mag_h, mag_v = compute_gradients(arr)
    pixel_w, pixel_h, col_breaks, row_breaks = detect_pixel_grid(
        mag_h, mag_v, edge_percentile
    )

    if pixel_w is None or pixel_h is None:
        if force_regular:
            print(
                "Error: --force-regular requested but period could not be estimated. "
                "Try adjusting --edge-percentile.",
                file=sys.stderr,
            )
            return
        art_w = len(_breaks_to_spans(col_breaks, W))
        art_h = len(_breaks_to_spans(row_breaks, H))
        print(f"  Period estimation failed; falling back to {art_w} x {art_h} detected spans")
    else:
        pixel_w /= scale
        pixel_h /= scale
        if square:
            avg = (pixel_w + pixel_h) / 2.0
            pixel_w = pixel_h = avg
        art_w = max(1, round(W / pixel_w))
        art_h = max(1, round(H / pixel_h))
        print(f"  Virtual pixel size : {pixel_w:.2f} x {pixel_h:.2f} px"
              + (f"  (scale {scale}x)" if scale != 1.0 else ""))
        print(f"  Expected art size  : {art_w} x {art_h} px")

    row_cv = _spacing_cv(row_breaks)
    col_cv = _spacing_cv(col_breaks)

    use_regular = (pixel_w is not None and pixel_h is not None) and not force_irregular
    if force_regular:
        use_regular = True
    if force_irregular:
        use_regular = False

    tile_suffix = ""
    if tile_size is not None:
        tw, th = tile_size
        n_cols = -(-W // tw)  # ceil division
        n_rows = -(-H // th)
        tile_suffix = f" tiled {n_cols}x{n_rows}"
        print(f"  Tiling: {n_cols} x {n_rows} tiles of {tw} x {th} px")

    if tile_size is not None:
        tw, th = tile_size
        strategy = ("regular" if use_regular else f"irregular span {sample_method}") + tile_suffix
        out_img = _downsample_tiled(
            arr, pixel_w, pixel_h, col_breaks, row_breaks,
            tw, th, edge_percentile, sample_method, use_regular, scale, square,
        )
    elif use_regular:
        strategy = "regular (nearest-neighbour at grid centers)"
        row_centers = _regular_centers(pixel_h, H, row_breaks)
        col_centers = _regular_centers(pixel_w, W, col_breaks)
        if abs(len(row_centers) - art_h) > 1:
            print(f"  Warning: regular grid yielded {len(row_centers)} rows, expected {art_h}")
        if abs(len(col_centers) - art_w) > 1:
            print(f"  Warning: regular grid yielded {len(col_centers)} cols, expected {art_w}")
        out_img = _downsample_regular(arr, row_centers, col_centers)
    else:
        strategy = f"irregular (span {sample_method} sampling)"
        row_spans = _breaks_to_spans(row_breaks, H)
        col_spans = _breaks_to_spans(col_breaks, W)
        if (scale != 1.0 or square) and pixel_h is not None:
            if scale >= 1.0:
                row_spans = _subdivide_spans(row_spans, pixel_h)
            else:
                row_spans = _merge_spans(row_spans, max(1, round(1.0 / scale)))
        if (scale != 1.0 or square) and pixel_w is not None:
            if scale >= 1.0:
                col_spans = _subdivide_spans(col_spans, pixel_w)
            else:
                col_spans = _merge_spans(col_spans, max(1, round(1.0 / scale)))
        if len(row_spans) != art_h:
            print(f"  Warning: {len(row_spans)} row spans detected, expected {art_h}")
        if len(col_spans) != art_w:
            print(f"  Warning: {len(col_spans)} col spans detected, expected {art_w}")
        out_img = _downsample_irregular(arr, row_spans, col_spans, sample_method)

    W_out, H_out = out_img.size
    compare_true_scale = max(1, round(min(W / W_out, H / H_out)))
    _print_report(W, H, W_out, H_out, row_cv, col_cv, strategy, compare_true_scale)

    print(f"\nSaving: {output_path}")
    out_img.save(str(output_path))

    if verbose:
        edges_img        = make_grid_overlay(img, col_breaks, row_breaks)
        compare_img      = make_compare(out_img, W, H)
        compare_true_img = make_compare_true(out_img, W, H)

        edges_path        = input_path.parent / (input_path.stem + f"_edges{flags_suffix}.png")
        compare_path      = input_path.parent / (input_path.stem + f"_compare{flags_suffix}.png")
        compare_true_path = input_path.parent / (input_path.stem + f"_compare_true{flags_suffix}.png")
        print(f"Saving: {edges_path}")
        edges_img.save(str(edges_path))
        print(f"Saving: {compare_path}")
        compare_img.save(str(compare_path))
        print(f"Saving: {compare_true_path}  ({compare_true_img.size[0]} x {compare_true_img.size[1]} px)")
        compare_true_img.save(str(compare_true_path))

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = docopt(__doc__)

    input_paths  = [Path(p) for p in args["<input>"]]
    output_arg   = args["--output"]
    edge_percentile = float(args["--edge-percentile"])
    sample_method   = args["--sample"]
    force_regular   = args["--force-regular"]
    force_irregular = args["--force-irregular"]
    square          = args["--square"]
    verbose         = args["--verbose"]
    scale           = float(args["--scale"])
    tile_size       = None
    if args["--tile"]:
        try:
            tile_size = _parse_tile_size(args["--tile"])
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if scale <= 0:
        print("Error: --scale must be a positive number", file=sys.stderr)
        sys.exit(1)

    if sample_method not in ("center", "mean", "median"):
        print("Error: --sample must be center, mean, or median", file=sys.stderr)
        sys.exit(1)

    if output_arg and len(input_paths) > 1:
        print("Error: --output cannot be used with multiple input files", file=sys.stderr)
        sys.exit(1)

    errors = [p for p in input_paths if not p.exists()]
    if errors:
        for p in errors:
            print(f"Error: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    suffix = _flags_suffix(force_regular, force_irregular, scale, tile_size, square) if verbose else ""

    for input_path in input_paths:
        output_path = Path(output_arg) if output_arg else (
            input_path.parent / (input_path.stem + f"_pixel{suffix}.png")
        )
        process_file(
            input_path, output_path,
            edge_percentile, sample_method,
            force_regular, force_irregular,
            scale, verbose,
            tile_size,
            suffix,
            square,
        )


if __name__ == "__main__":
    main()
