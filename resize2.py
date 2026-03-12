#!/usr/bin/env python3
"""Downsample AI-generated pixel art using improved v2 grid detection.

Uses multi-threshold consensus, adaptive gap-filling, and 2D strip refinement
for more robust grid detection.  Boundary-trimmed sampling reduces colour
bleed from anti-aliased virtual-pixel edges.

Otherwise identical to resize.py -- all CLI options are the same.

Usage:
    resize2.py <input>... [--output=PATH] [-p P] [--sample=METHOD] [-g R]
               [-r | -i] [-s S] [-t SIZE] [-q] [-v]
    resize2.py -h | --help

Arguments:
    <input>                 Input image path(s).

Options:
    -o PATH --output=PATH       Output image path (default: <input>_pixel.png).
    -p P --edge-percentile=P    Gradient percentile threshold for break detection [default: 85].
    --sample=METHOD             Sampling method for irregular grids: center, mean, median [default: center].
    -g R --max-gap=R            Max gap ratio before subdividing [default: 1.6].
    -r --force-regular          Force regular-grid strategy.
    -i --force-irregular        Force irregular span-based strategy.
    -s S --scale=S              Divide detected virtual pixel size by S [default: 1.0].
    -t SIZE --tile=SIZE         Split image into tiles for local phase correction.
    -q --square                 Force square virtual pixels.
    -v --verbose                Also save _edges.png, _compare.png, and _compare_true.png.
    -h --help                   Show this screen.
"""

import resize
from detect2 import detect_pixel_grid_v2


# ---------------------------------------------------------------------------
# Patch 1: Replace grid detection with v2 consensus pipeline
# ---------------------------------------------------------------------------

resize.detect_pixel_grid = detect_pixel_grid_v2


# ---------------------------------------------------------------------------
# Patch 2: Boundary-trimmed sampling
#
# AI generators often anti-alias virtual-pixel edges, blending colours from
# neighbouring pixels.  Trimming the outer ~15 % of each block before
# sampling avoids pulling in these blended boundary colours.
# ---------------------------------------------------------------------------

_TRIM = 0.15

_original_sample_block = resize._sample_block


def _sample_block_trimmed(arr, r0, r1, c0, c1, method):
    h, w = r1 - r0, c1 - c0
    if h >= 5:
        tr = max(1, int(h * _TRIM))
        r0, r1 = r0 + tr, r1 - tr
    if w >= 5:
        tc = max(1, int(w * _TRIM))
        c0, c1 = c0 + tc, c1 - tc
    return _original_sample_block(arr, r0, r1, c0, c1, method)


resize._sample_block = _sample_block_trimmed


# ---------------------------------------------------------------------------
# Entry point -- delegates to resize.main() with patches applied
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use this module's docstring for CLI help
    resize.__doc__ = __doc__
    resize.main()
