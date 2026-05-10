"""Derive tile-grid parity fixtures from upstream Lfm2VlImageProcessorFast.

Run:
    python3 scripts/derive_tile_grid_fixtures.py

This prints the expected (rows, cols, tile_h, tile_w, thumbnail) for each
test case used in src/preproc/tile_grid.rs::tests::pick_tile_grid_parity_cases.

Upstream module:
    transformers/models/lfm2_vl/image_processing_lfm2_vl_fast.py
"""
import math
import sys

sys.path.insert(0, '/Users/user/Library/Python/3.9/lib/python/site-packages')

from transformers.models.lfm2_vl.image_processing_lfm2_vl_fast import (
    Lfm2VlImageProcessorFast,
    find_closest_aspect_ratio,
)

proc = Lfm2VlImageProcessorFast()

print(f"Upstream defaults:")
print(f"  encoder_patch_size={proc.encoder_patch_size}")
print(f"  downsample_factor={proc.downsample_factor}")
print(f"  tile_size={proc.tile_size}")
print(f"  min_tiles={proc.min_tiles}, max_tiles={proc.max_tiles}")
print(f"  min_image_tokens={proc.min_image_tokens}, max_image_tokens={proc.max_image_tokens}")
print(f"  max_pixels_tolerance={proc.max_pixels_tolerance}")
print(f"  use_thumbnail={proc.use_thumbnail}")
print()


def expected(w, h, budget=None):
    b = budget or {}
    min_tiles = b.get('min_tiles', proc.min_tiles)
    max_tiles = b.get('max_tiles', proc.max_tiles)
    min_image_tokens = b.get('min_image_tokens', proc.min_image_tokens)
    max_image_tokens = b.get('max_image_tokens', proc.max_image_tokens)
    use_thumbnail = b.get('use_thumbnail', proc.use_thumbnail)
    tolerance = b.get('max_pixels_tolerance', proc.max_pixels_tolerance)

    too_large = proc._is_image_too_large(
        h, w, max_image_tokens,
        proc.encoder_patch_size, proc.downsample_factor, tolerance
    )
    new_w, new_h = proc.smart_resize(
        h, w, proc.downsample_factor,
        min_image_tokens, max_image_tokens, proc.encoder_patch_size
    )

    if too_large:
        # multi-tile path
        ratios = proc._target_ratios(min_tiles, max_tiles)
        # find_closest_aspect_ratio returns (grid_width, grid_height)
        # crop_image_to_patches returns (images, grid_width, grid_height)
        # the caller assigns: images, num_rows, num_cols = crop_image_to_patches(...)
        # So num_rows = grid_width, num_cols = grid_height
        grid_width, grid_height = find_closest_aspect_ratio(
            w / h, ratios, w, h, proc.tile_size
        )
        thumb = (new_h, new_w) if use_thumbnail else None
        # Our Rust TileGrid::new(rows, cols, tile_h, tile_w, thumbnail)
        # rows = grid_width (upstream num_rows)
        # cols = grid_height (upstream num_cols)
        # But wait — let's double-check the semantics...
        # _get_grid_layout returns (grid_width, grid_height, target_width, target_height, total)
        # target_width = tile_size * grid_width
        # target_height = tile_size * grid_height
        # F.resize(image, (target_height, target_width)) → (H=target_height, W=target_width)
        # So grid_width tiles fit along width, grid_height tiles along height
        # "rows" = number of vertical strips = grid_height
        # "cols" = number of horizontal strips = grid_width
        # But caller does: images, num_rows, num_cols = crop_image_to_patches(...)
        # and crop_image_to_patches returns (..., grid_width, grid_height)
        # So num_rows=grid_width, num_cols=grid_height from the caller's perspective
        # That means upstream's "num_rows" is actually the width-direction count (confusing!)
        # Our Rust uses rows=height-tiles, cols=width-tiles semantics
        # For 1920x1080 (wide image), we expect cols>rows (more columns than rows)
        # aspect_ratio = w/h = 1920/1080 ≈ 1.778
        # best ratio: find_closest_aspect_ratio with aspect 1.778
        # (2,4): ratio[0]/ratio[1] = 2/4 = 0.5 (nope)
        # (4,2): ratio[0]/ratio[1] = 4/2 = 2.0 (closer to 1.778)
        # So for wide image, grid_width=4, grid_height=2
        # target_width = 512*4=2048, target_height = 512*2=1024 → makes sense for wide image
        # So grid_width corresponds to cols (horizontal tiles)
        # upstream num_rows=grid_width=4 is actually number of COLUMNS
        # This is the naming confusion in upstream
        # For our Rust: rows=grid_height (height-direction), cols=grid_width (width-direction)
        rows_rust = grid_height  # vertical tiles
        cols_rust = grid_width   # horizontal tiles
        return rows_rust, cols_rust, proc.tile_size, proc.tile_size, thumb
    else:
        # single-tile path
        return 1, 1, new_h, new_w, None


# Helper to verify rows/cols mapping for 1920x1080
print("=== Semantic verification: 1920x1080 (wide, 16:9) ===")
w, h = 1920, 1080
ratios = proc._target_ratios(proc.min_tiles, proc.max_tiles)
gw, gh = find_closest_aspect_ratio(w/h, ratios, w, h, proc.tile_size)
print(f"  find_closest_aspect_ratio(1920/1080) -> grid_width={gw}, grid_height={gh}")
print(f"  target_width={proc.tile_size*gw}, target_height={proc.tile_size*gh}")
print(f"  (expected: target_width > target_height for wide image)")
print()

print("=== All target_ratios (min=2, max=10) ===")
print(f"  {ratios[:10]}...")
print()

# ── All 28 test cases ─────────────────────────────────────────────────────
cases = [
    # (w, h, budget_override, label)
    # Single-tile path
    (256, 256, {}, "small_square_256"),
    (512, 512, {}, "small_square_512"),
    (723, 724, {}, "just_below_threshold_723x724"),
    (32, 32, {}, "tiny_32x32"),
    (320, 240, {}, "4:3_within_budget_320x240"),
    (384, 216, {}, "16:9_within_budget_384x216"),
    (640, 480, {}, "4:3_above_max_area_640x480"),
    (480, 270, {}, "16:9_below_threshold_480x270"),
    # Pathological aspects
    (32, 1024, {}, "pathological_32x1024"),
    (1024, 32, {}, "pathological_1024x32"),
    (1, 8000, {}, "pathological_1x8000"),
    (8000, 1, {}, "pathological_8000x1"),
    # Multi-tile path
    (1024, 1024, {}, "multi_1024x1024"),
    (768, 768, {}, "multi_768x768"),
    (1920, 1080, {}, "multi_1920x1080"),
    (1080, 1920, {}, "multi_1080x1920"),
    (1280, 720, {}, "multi_1280x720"),
    (2560, 1440, {}, "multi_2560x1440"),
    (1440, 2560, {}, "multi_1440x2560"),
    # Aspect-ratio tie-break pairs
    (1024, 768, {}, "tie_1024x768"),
    (768, 1024, {}, "tie_768x1024"),
    # 2x1 vs 1x2 boundary
    (1600, 800, {}, "boundary_1600x800"),
    (800, 1600, {}, "boundary_800x1600"),
    # Just above threshold
    (720, 730, {}, "just_above_threshold_720x730"),
    # Non-default budgets
    (256, 256, {'min_image_tokens': 32, 'max_image_tokens': 64, 'min_tiles': 2, 'max_tiles': 4, 'use_thumbnail': False}, "fast_256x256"),
    (1024, 1024, {'min_image_tokens': 32, 'max_image_tokens': 64, 'min_tiles': 2, 'max_tiles': 4, 'use_thumbnail': False}, "fast_1024x1024"),
    (1920, 1080, {'min_tiles': 4, 'max_tiles': 10}, "min4_1920x1080"),
    (1080, 1920, {'min_tiles': 4, 'max_tiles': 10}, "min4_1080x1920"),
]

print("=== Fixture derivation ===")
for w, h, budget, label in cases:
    rows, cols, tile_h, tile_w, thumb = expected(w, h, budget)
    thumb_str = f"Some(({thumb[0]}, {thumb[1]}))" if thumb else "None"
    print(f"  // {label}")
    print(f"  ({w}, {h}, budget, TileGrid::new({rows}, {cols}, {tile_h}, {tile_w}, {thumb_str})),")
    print()
