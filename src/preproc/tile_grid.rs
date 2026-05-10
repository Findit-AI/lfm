//! Tile-grid algorithm port of upstream `image_processing_lfm2_vl.py`.
//! Two paths:
//! - **Multi-tile**: uniform 512×512 main tiles via `find_closest_aspect_ratio`
//!   + optional thumbnail dynamically sized via `smart_resize`.
//! - **Single-tile**: dynamically sized via `smart_resize`.
//!
//! Per spec §8.3.

use crate::{
  error::{Error, Result},
  options::ImageBudget,
};

/// Constants per LFM2.5-VL preprocessor_config.json.
pub(crate) const PATCH_SIZE: u32 = 16;
pub(crate) const DOWNSAMPLE_FACTOR: u32 = 2;
pub(crate) const TILE_PIXEL_UNIT: u32 = PATCH_SIZE * DOWNSAMPLE_FACTOR; // 32
pub(crate) const FULL_TILE_SIZE: u32 = 512;

/// One image's tile-grid layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
  /// Main-tile grid rows (1 in single-tile path).
  rows: u32,
  /// Main-tile grid cols (1 in single-tile path).
  cols: u32,
  /// Main-tile height in pixels (always 512 in multi-tile path; dynamic in single-tile).
  tile_h: u32,
  /// Main-tile width in pixels (always 512 in multi-tile path; dynamic in single-tile).
  tile_w: u32,
  /// `Some((thumb_h, thumb_w))` only in multi-tile + use_thumbnail.
  thumbnail: Option<(u32, u32)>,
}

impl TileGrid {
  /// Construct a new `TileGrid`.
  pub const fn new(
    rows: u32,
    cols: u32,
    tile_h: u32,
    tile_w: u32,
    thumbnail: Option<(u32, u32)>,
  ) -> Self {
    Self {
      rows,
      cols,
      tile_h,
      tile_w,
      thumbnail,
    }
  }

  /// Main-tile grid rows (1 in single-tile path).
  pub const fn rows(&self) -> u32 {
    self.rows
  }

  /// Main-tile grid cols (1 in single-tile path).
  pub const fn cols(&self) -> u32 {
    self.cols
  }

  /// Main-tile height in pixels (always 512 in multi-tile path; dynamic in single-tile).
  pub const fn tile_h(&self) -> u32 {
    self.tile_h
  }

  /// Main-tile width in pixels (always 512 in multi-tile path; dynamic in single-tile).
  pub const fn tile_w(&self) -> u32 {
    self.tile_w
  }

  /// `Some((thumb_h, thumb_w))` only in multi-tile + use_thumbnail.
  pub const fn thumbnail(&self) -> Option<(u32, u32)> {
    self.thumbnail
  }

  /// Set rows.
  pub fn set_rows(&mut self, rows: u32) {
    self.rows = rows;
  }

  /// Set cols.
  pub fn set_cols(&mut self, cols: u32) {
    self.cols = cols;
  }

  /// Set tile height.
  pub fn set_tile_h(&mut self, tile_h: u32) {
    self.tile_h = tile_h;
  }

  /// Set tile width.
  pub fn set_tile_w(&mut self, tile_w: u32) {
    self.tile_w = tile_w;
  }

  /// Set thumbnail.
  pub fn set_thumbnail(&mut self, thumbnail: Option<(u32, u32)>) {
    self.thumbnail = thumbnail;
  }

  /// Builder: set rows (chainable).
  pub const fn with_rows(mut self, rows: u32) -> Self {
    self.rows = rows;
    self
  }

  /// Builder: set cols (chainable).
  pub const fn with_cols(mut self, cols: u32) -> Self {
    self.cols = cols;
    self
  }

  /// Builder: set tile height (chainable).
  pub const fn with_tile_h(mut self, tile_h: u32) -> Self {
    self.tile_h = tile_h;
    self
  }

  /// Builder: set tile width (chainable).
  pub const fn with_tile_w(mut self, tile_w: u32) -> Self {
    self.tile_w = tile_w;
    self
  }

  /// Builder: set thumbnail (chainable).
  pub const fn with_thumbnail(mut self, thumbnail: Option<(u32, u32)>) -> Self {
    self.thumbnail = thumbnail;
    self
  }

  /// Total number of tiles (main + thumbnail if present).
  pub const fn num_tiles(&self) -> usize {
    (self.rows as usize) * (self.cols as usize)
      + match self.thumbnail {
        Some(_) => 1,
        None => 0,
      }
  }

  /// Total `<image>` tokens this grid will expand to in the prompt.
  ///
  /// Multi-tile path: `rows × cols × tokens_per_main_tile + thumbnail_tokens`.
  /// Single-tile path: just `tokens_per_main_tile` (no thumbnail).
  /// `tokens_per_full_tile = (tile_h / TILE_PIXEL_UNIT) × (tile_w / TILE_PIXEL_UNIT)`.
  ///
  /// Used by `generate()` for the cheap header-only admission check —
  /// we get exact image-token counts before paying for full image
  /// decode + smart_resize + flatten_to_patches.
  pub const fn num_image_tokens(&self) -> usize {
    let main = (self.rows as usize)
      * (self.cols as usize)
      * (self.tile_h / TILE_PIXEL_UNIT) as usize
      * (self.tile_w / TILE_PIXEL_UNIT) as usize;
    let thumb = match self.thumbnail {
      Some((th, tw)) => (th / TILE_PIXEL_UNIT) as usize * (tw / TILE_PIXEL_UNIT) as usize,
      None => 0,
    };
    main + thumb
  }

  /// Build the chat-template `ImagePlaceholderInfo` for this grid
  /// without going through `flatten_to_patches`. Lets `generate()`
  /// render the full prompt (including image-marker expansion) from
  /// header-only image dimensions, BEFORE doing any expensive image
  /// decode + patchify. The actual pixel preprocessing then runs
  /// per-image inside the vision-encode loop, so peak memory is
  /// bounded by ONE image's pixel_values at a time instead of the
  /// whole batch's.
  pub const fn to_placeholder_info(&self) -> crate::chat_template::ImagePlaceholderInfo {
    let tokens_per_main_tile =
      (self.tile_h / TILE_PIXEL_UNIT) as usize * (self.tile_w / TILE_PIXEL_UNIT) as usize;
    let thumbnail_tokens = match self.thumbnail {
      Some((th, tw)) => Some((th / TILE_PIXEL_UNIT) as usize * (tw / TILE_PIXEL_UNIT) as usize),
      None => None,
    };
    crate::chat_template::ImagePlaceholderInfo::new(
      self.rows as usize,
      self.cols as usize,
      tokens_per_main_tile,
      thumbnail_tokens,
    )
  }
}

/// Pick a tile grid for the given source image dims under the budget.
///
/// **TODO (v0.1):** capture upstream `Lfm2VlImageProcessor` fixture for
/// 20+ (src_w, src_h, ImageBudget) → grid cases. Current tests exercise
/// constructive properties (path selection, dim multiples, monotonicity).
pub fn pick_tile_grid(src_w: u32, src_h: u32, budget: &ImageBudget) -> Result<TileGrid> {
  if src_w == 0 || src_h == 0 {
    return Err(Error::ImageTooSmall {
      w: src_w,
      h: src_h,
      min_w: TILE_PIXEL_UNIT,
      min_h: TILE_PIXEL_UNIT,
    });
  }
  budget.validate()?;

  // Fix 2: use upstream _is_image_too_large which rounds dims to total_factor first,
  // then compares the ROUNDED product to the tolerance-scaled threshold.
  // Ports upstream `_is_image_too_large` from
  // transformers/models/lfm2_vl/image_processing_lfm2_vl_fast.py lines 343-355.
  //
  // do_image_splitting gate: upstream resize_and_split (line 376) sets
  // `do_image_splitting = not (min_tiles == max_tiles == 1)` and routes
  // to the multi-tile path only when `is_image_large AND
  // do_image_splitting`. Without this gate, a custom budget with
  // min_tiles=max_tiles=1 would: (a) enter the multi-tile branch on
  // any large image, (b) get rows=cols=1 from find_closest_aspect_ratio
  // (only candidate), (c) force a 512×512 main tile + thumbnail when
  // use_thumbnail is true. build_image_block then treats rows=1 cols=1
  // as the single-tile format and emits NO thumbnail marker, while
  // PreprocessedImage's tile count would still include the thumbnail —
  // silent prompt/feature divergence.
  let do_image_splitting = !(budget.min_tiles() == 1 && budget.max_tiles() == 1);

  if is_image_too_large(
    src_w,
    src_h,
    budget.max_image_tokens(),
    budget.max_pixels_tolerance(),
  ) && do_image_splitting
  {
    // ===== Multi-tile =====
    // Fix 4: use corrected find_closest_aspect_ratio (set + sort + tie-break).
    let (rows, cols) = find_closest_aspect_ratio(
      src_w as f64 / src_h as f64,
      budget.min_tiles() as u32,
      budget.max_tiles() as u32,
      src_w,
      src_h,
    );
    // Upstream gates the thumbnail on the selected grid having more than
    // one tile (transformers/models/lfm2_vl/image_processing_lfm2_vl_fast.py
    // line 301: `if use_thumbnail and grid_width * grid_height != 1`).
    // Without this check, a budget like `min_tiles=1, max_tiles=2` on a
    // large square image picks rows=cols=1 in find_closest_aspect_ratio
    // (square aspect best matches square candidate) and we attached a
    // thumbnail anyway. PreprocessedImage's tile count would include
    // the thumbnail features, but build_image_block treats rows=cols=1
    // as the SINGLE-tile format and emits NO `<|img_thumbnail|>` marker.
    // Token count still matches → ImageTokenCountMismatch never fires →
    // model conditioning silently corrupted.
    let thumbnail = if budget.use_thumbnail() && rows * cols != 1 {
      let (tw, th) = smart_resize(
        src_w,
        src_h,
        budget.min_image_tokens(),
        budget.max_image_tokens(),
      );
      // Hard cap: smart_resize can produce >max_image_tokens output for
      // extreme aspect ratios (e.g., 1×100000 yields ~32×161888 = 5059
      // tokens vs a 256 budget). flatten_to_patches would then pad every
      // tile to the huge thumbnail patch count and allocate hundreds of
      // MB of pixel_values — fail closed instead.
      if smart_resize_tokens(tw, th) as usize > budget.max_image_tokens() {
        return Err(Error::TileGridImpossible {
          w: src_w,
          h: src_h,
          budget: *budget,
        });
      }
      Some((th, tw))
    } else {
      None
    };
    Ok(TileGrid::new(
      rows,
      cols,
      FULL_TILE_SIZE,
      FULL_TILE_SIZE,
      thumbnail,
    ))
  } else {
    // ===== Single-tile =====
    // Fix 3: use corrected smart_resize.
    let (tw, th) = smart_resize(
      src_w,
      src_h,
      budget.min_image_tokens(),
      budget.max_image_tokens(),
    );
    // Same hard cap as the thumbnail branch — extreme aspect ratios
    // can blow past max_image_tokens here too.
    if smart_resize_tokens(tw, th) as usize > budget.max_image_tokens() {
      return Err(Error::TileGridImpossible {
        w: src_w,
        h: src_h,
        budget: *budget,
      });
    }
    Ok(TileGrid::new(1, 1, th, tw, None))
  }
}

/// Ports upstream `_is_image_too_large` from
/// transformers/models/lfm2_vl/image_processing_lfm2_vl_fast.py lines 343-355.
///
/// Key behavior: rounds dims to `total_factor` (=32) first using
/// `round_by_factor`, then clamps to `>= PATCH_SIZE` (=16, NOT total_factor),
/// then compares the ROUNDED product to the tolerance-scaled threshold.
fn is_image_too_large(src_w: u32, src_h: u32, max_image_tokens: usize, tolerance: f32) -> bool {
  // total_factor = encoder_patch_size * downsample_factor = 32
  let total_factor = TILE_PIXEL_UNIT as f64; // 32
  // Note: clamp uses PATCH_SIZE (=16), NOT total_factor (=32) — intentional
  // asymmetry in upstream between _is_image_too_large and smart_resize.
  let patch_size = PATCH_SIZE; // 16
  let h_bar = (patch_size).max(round_by_factor(src_h as f64, total_factor));
  let w_bar = (patch_size).max(round_by_factor(src_w as f64, total_factor));
  // threshold = max_image_tokens * encoder_patch_size^2 * downsample_factor^2 * tolerance
  //           = max_image_tokens * PATCH_SIZE^2 * DOWNSAMPLE_FACTOR^2 * tolerance
  //           = max_image_tokens * 1024 * tolerance   (since 16^2 * 2^2 = 1024)
  let threshold = (max_image_tokens as f64)
    * (PATCH_SIZE as f64).powi(2)
    * (DOWNSAMPLE_FACTOR as f64).powi(2)
    * (tolerance as f64);
  (h_bar as u64) * (w_bar as u64) > threshold as u64
}

/// Round `number` to the nearest integer divisible by `factor`.
/// Ports upstream `round_by_factor` (line 49-51).
/// Uses round-half-to-even (banker's rounding) to match Python's `round()`.
fn round_by_factor(number: f64, factor: f64) -> u32 {
  // Python's round() uses banker's rounding (round half to even).
  // Rust's f64::round() uses round-half-away-from-zero.
  // We need to match Python for correctness.
  let q = number / factor;
  // Banker's rounding: round half to even
  let floored = q.floor();
  let frac = q - floored;
  let rounded = if (frac - 0.5).abs() < f64::EPSILON {
    // exactly 0.5: round to even
    if (floored as i64) % 2 == 0 {
      floored
    } else {
      floored + 1.0
    }
  } else {
    q.round()
  };
  (rounded as u32) * (factor as u32)
}

/// Enumerate (rows, cols) candidates with `rows*cols ∈ [min_tiles, max_tiles]`,
/// pick the one whose aspect (cols/rows) is closest to `src_aspect`.
/// Ports upstream `find_closest_aspect_ratio` from
/// transformers/models/lfm2_vl/image_processing_lfm2_vl_fast.py lines 54-92
/// (including the `_target_ratios` set-then-sort pattern from lines 231-240).
///
/// Upstream `_target_ratios` enumerates ALL (w, h) pairs (not just divisor pairs)
/// such that `min_tiles <= w*h <= max_tiles`, deduplicates via a set, then sorts
/// by `w*h` ascending. `find_closest_aspect_ratio` then iterates that sorted list
/// and applies:
///   - `ratio_diff < best_ratio_diff` → always update best
///   - `ratio_diff == best_ratio_diff` → update if `area > 0.5 * target_area`
///
/// In upstream, ratio = (grid_width, grid_height); here we use (cols, rows) for
/// the same concept (cols = horizontal tiles, rows = vertical tiles). The returned
/// `(rows, cols)` corresponds to upstream's `(grid_height, grid_width)`.
fn find_closest_aspect_ratio(
  src_aspect: f64,
  min_tiles: u32,
  max_tiles: u32,
  src_w: u32,
  src_h: u32,
) -> (u32, u32) {
  // the previous in-place build-and-sort
  // approach iterated candidates in Rust insertion order within each
  // product class, which differs from CPython's `sorted(set(ratios),
  // key=product)` order. The tie-break at the end of this function is
  // ITERATION-ORDER-SENSITIVE for equal-product equal-diff candidates
  // (target_area is class-wide so the `area > 0.5 * target_area` check is
  // uniform — the LAST candidate wins when true, the FIRST wins when
  // false). Different orders → different picks. Concrete diverging case:
  // 1280×512 with min=max=4 → upstream picks (2,2), Rust picked (4,1).
  //
  // Fix: use a precomputed static lookup that mirrors CPython's set
  // iteration order bit-for-bit. See `target_ratios::target_ratios` and
  // `scripts/derive_target_ratios.py`.
  let candidates = super::target_ratios::target_ratios(min_tiles, max_tiles);

  let src_area = (src_w as u64) * (src_h as u64);
  let mut best_ratio_diff = f64::INFINITY;
  // (cols, rows) in our terms — upstream's (grid_width, grid_height)
  let mut best: Option<(u32, u32)> = None;

  for &(cols, rows) in candidates {
    // upstream: target_aspect_ratio = ratio[0] / ratio[1] = cols / rows
    let target_aspect = (cols as f64) / (rows as f64);
    let ratio_diff = (src_aspect - target_aspect).abs();

    if ratio_diff < best_ratio_diff {
      best_ratio_diff = ratio_diff;
      best = Some((cols, rows));
    } else if ratio_diff == best_ratio_diff {
      // Upstream tie-break: prefer if area > 0.5 * target_area
      // target_area = image_size^2 * ratio[0] * ratio[1] = tile_size^2 * cols * rows
      let target_area =
        (FULL_TILE_SIZE as u64) * (FULL_TILE_SIZE as u64) * (cols as u64) * (rows as u64);
      if src_area * 2 > target_area {
        best = Some((cols, rows));
      }
    }
  }

  // best = (cols, rows) in our terms; return (rows, cols) to match TileGrid convention.
  let (cols, rows) = best.unwrap_or((1, 1));
  (rows, cols)
}

/// Convert a `smart_resize` output to a token count
/// (`(h / TILE_PIXEL_UNIT) * (w / TILE_PIXEL_UNIT)`).
fn smart_resize_tokens(w: u32, h: u32) -> u32 {
  (w / TILE_PIXEL_UNIT) * (h / TILE_PIXEL_UNIT)
}

/// Resize (src_w, src_h) to fit pixel budget, preserve aspect, snap to TILE_PIXEL_UNIT.
/// Returns (width, height).
///
/// Ports upstream `smart_resize` from
/// transformers/models/lfm2_vl/image_processing_lfm2_vl_fast.py lines 310-341.
///
/// Key behaviors:
/// - Initial round-to-nearest sets h_bar/w_bar (using `round_by_factor`)
/// - Threshold compares ROUNDED product, not raw area
/// - Shrink path: `floor(dim / beta / total_factor) * total_factor`, max-clamped to total_factor
/// - Grow path: `ceil(dim * beta / total_factor) * total_factor`, no max-clamp
/// - Returns (w_bar, h_bar) — width first (matches our (w, h) return convention)
///
/// **Pathological extreme-aspect note:** for inputs like 1×100000, the
/// initial round produces h_bar=100000, w_bar=32 (rounded-area
/// 3.2M >> max_pixels=262144). The shrink branch then computes
/// `beta = sqrt(raw_area / max_pixels)` from RAW area (not rounded),
/// so beta < 1 even though we're trying to shrink — yielding
/// h_bar = floor(h / beta / 32) * 32 = ~161888 (much LARGER than the
/// initial h_bar). This is a faithful port of upstream's behavior;
/// `pick_tile_grid` enforces a hard token cap on the result and
/// returns `TileGridImpossible` if the output would exceed the budget,
/// rather than silently producing a multi-MB pixel_values tensor.
fn smart_resize(src_w: u32, src_h: u32, min_tokens: usize, max_tokens: usize) -> (u32, u32) {
  let total_factor = TILE_PIXEL_UNIT as f64; // 32.0
  let min_pixels =
    (min_tokens as f64) * (PATCH_SIZE as f64).powi(2) * (DOWNSAMPLE_FACTOR as f64).powi(2);
  let max_pixels =
    (max_tokens as f64) * (PATCH_SIZE as f64).powi(2) * (DOWNSAMPLE_FACTOR as f64).powi(2);

  // Initial round: snap to nearest multiple of total_factor, clamp to >= total_factor.
  let mut h_bar = (TILE_PIXEL_UNIT).max(round_by_factor(src_h as f64, total_factor));
  let mut w_bar = (TILE_PIXEL_UNIT).max(round_by_factor(src_w as f64, total_factor));

  let rounded_area = (h_bar as f64) * (w_bar as f64);

  if rounded_area > max_pixels {
    // SHRINK: beta = sqrt(raw_area / max_pixels), then floor each dim.
    let raw_area = (src_w as f64) * (src_h as f64);
    let beta = (raw_area / max_pixels).sqrt();
    // floor(dim / beta / total_factor) * total_factor, clamped to >= total_factor
    h_bar =
      TILE_PIXEL_UNIT.max(((src_h as f64 / beta / total_factor).floor() as u32) * TILE_PIXEL_UNIT);
    w_bar =
      TILE_PIXEL_UNIT.max(((src_w as f64 / beta / total_factor).floor() as u32) * TILE_PIXEL_UNIT);
  } else if rounded_area < min_pixels {
    // GROW: beta = sqrt(min_pixels / raw_area), then ceil each dim. NO max-clamp.
    let raw_area = (src_w as f64) * (src_h as f64);
    let beta = (min_pixels / raw_area).sqrt();
    // ceil(dim * beta / total_factor) * total_factor
    h_bar = ((src_h as f64 * beta / total_factor).ceil() as u32) * TILE_PIXEL_UNIT;
    w_bar = ((src_w as f64 * beta / total_factor).ceil() as u32) * TILE_PIXEL_UNIT;
  }

  (w_bar, h_bar)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn small_square_routes_single_tile() {
    let g = pick_tile_grid(256, 256, &ImageBudget::new()).unwrap();
    assert_eq!((g.rows(), g.cols()), (1, 1));
    assert!(g.thumbnail().is_none());
  }

  #[test]
  fn large_square_routes_multi_tile() {
    let g = pick_tile_grid(1024, 1024, &ImageBudget::new()).unwrap();
    assert!(g.rows() * g.cols() >= 2);
    assert_eq!(g.tile_h(), FULL_TILE_SIZE);
    assert_eq!(g.tile_w(), FULL_TILE_SIZE);
    assert!(g.thumbnail().is_some());
  }

  #[test]
  fn smart_resize_snaps_to_unit_multiples() {
    let (w, h) = smart_resize(1920, 1080, 64, 256);
    assert_eq!(w % TILE_PIXEL_UNIT, 0);
    assert_eq!(h % TILE_PIXEL_UNIT, 0);
  }

  #[test]
  fn aspect_picker_prefers_correct_ratio() {
    // Wide image (3:1) with min=2, max=10: should prefer cols/rows=3 closest to src=3.0.
    let (rows, cols) = find_closest_aspect_ratio(3.0, 2, 10, 1024 * 9, 1024);
    assert_eq!(cols as f64 / rows as f64, 3.0);
  }

  #[test]
  fn zero_dimension_image_rejects() {
    let r = pick_tile_grid(0, 100, &ImageBudget::new());
    assert!(matches!(r, Err(Error::ImageTooSmall { .. })));
  }

  #[test]
  fn extreme_aspect_ratios_reject_with_tile_grid_impossible() {
    // 1×100000 (and its transpose) — smart_resize's shrink branch
    // produces ~32×161888 = 5059 image tokens, far above the 256
    // budget. Without the hard cap, flatten_to_patches would pad
    // every tile to that size and allocate >5 MB of pixel_values.
    // Now we fail closed with TileGridImpossible.
    for (w, h) in [(1u32, 100_000u32), (100_000, 1), (1, 50_000), (50_000, 1)] {
      let r = pick_tile_grid(w, h, &ImageBudget::new());
      assert!(
        matches!(r, Err(Error::TileGridImpossible { .. })),
        "{w}x{h} must reject with TileGridImpossible, got {r:?}"
      );
    }
  }

  #[test]
  fn to_placeholder_info_matches_preprocessed_image() {
    // TileGrid::to_placeholder_info()
    // must produce the same ImagePlaceholderInfo that the full
    // preprocessing path (PreprocessedImage::to_placeholder_info)
    // would, since generate() now derives placeholders from grids
    // alone (deferring full preprocessing to inside the vision-encode
    // loop). If these drift, the rendered prompt's <image> token
    // count won't match what flatten_to_patches actually produces and
    // splice() would corrupt embeddings.
    //
    // Multi-tile case: 1024×1024 default budget → multi-tile + thumb.
    let grid = pick_tile_grid(1024, 1024, &ImageBudget::new()).unwrap();
    let info = grid.to_placeholder_info();
    assert_eq!(
      info.num_image_tokens(),
      grid.num_image_tokens(),
      "ImagePlaceholderInfo + TileGrid token counts must match"
    );

    // Single-tile case: 256×256 default budget → single-tile.
    let grid_s = pick_tile_grid(256, 256, &ImageBudget::new()).unwrap();
    let info_s = grid_s.to_placeholder_info();
    assert_eq!(info_s.num_image_tokens(), grid_s.num_image_tokens());
    assert_eq!(info_s.thumbnail_tokens(), None);

    // Wide multi-tile: 1920x1080.
    let grid_w = pick_tile_grid(1920, 1080, &ImageBudget::new()).unwrap();
    let info_w = grid_w.to_placeholder_info();
    assert_eq!(info_w.num_image_tokens(), grid_w.num_image_tokens());
  }

  #[test]
  fn upstream_set_iteration_order_drives_tie_break() {
    // with min_tiles=max_tiles=4 the
    // product=4 candidates {(1,4), (4,1), (2,2)} all have the same
    // tie-break target_area. Upstream's CPython set iteration order is
    // [(4,1), (1,4), (2,2)] for this set. For src 1280×512 (aspect 2.5,
    // area 655360 > 0.5*4*512² = 524288), the tie-break is TRUE so the
    // LAST candidate in iteration order wins → upstream picks (2,2).
    //
    // Before the static-table fix, our Rust code's natural insertion
    // order was [(1,4), (2,2), (4,1)] and we picked (4,1) — same tile
    // count as upstream but a DIFFERENT crop layout. Model conditioning
    // would diverge silently from what upstream produces.
    //
    // Verify: result is (rows=2, cols=2), matching upstream.
    let (rows, cols) = find_closest_aspect_ratio(2.5, 4, 4, 1280, 512);
    assert_eq!(
      (rows, cols),
      (2, 2),
      "upstream picks (2,2) for 1280×512 / min=max=4; ours must too"
    );

    // Transpose: 512×1280 (aspect 0.4). No tie here — diffs from 0.4:
    // (4,1)=3.6, (1,4)=0.15, (2,2)=0.6 → (1,4) wins outright. In our
    // (rows, cols) convention that's (rows=4, cols=1).
    let (rows_t, cols_t) = find_closest_aspect_ratio(0.4, 4, 4, 512, 1280);
    assert_eq!((rows_t, cols_t), (4, 1));

    // Larger 2.5:1: same tie pattern as 1280×512.
    let (rows_l, cols_l) = find_closest_aspect_ratio(2.5, 4, 4, 3200, 1280);
    assert_eq!((rows_l, cols_l), (2, 2));
  }

  #[test]
  fn one_by_one_grid_in_multi_tile_branch_attaches_no_thumbnail() {
    // with min_tiles=1, max_tiles=2,
    // a large square image is `is_image_too_large=true`,
    // `do_image_splitting=true`, and `find_closest_aspect_ratio` picks
    // (rows=1, cols=1) (square aspect best matches square candidate).
    // Before the fix, the multi-tile branch attached a thumbnail
    // anyway; build_image_block then treated rows=cols=1 as the
    // SINGLE-tile format and emitted no `<|img_thumbnail|>` marker —
    // silent prompt/feature divergence (token count still matched).
    // Upstream gates the thumbnail on grid_width*grid_height != 1; we
    // now do the same.
    let mut budget = ImageBudget::new();
    budget.set_min_tiles(1);
    budget.set_max_tiles(2);
    let g = pick_tile_grid(1024, 1024, &budget).expect("budget is valid");
    assert_eq!(
      (g.rows(), g.cols()),
      (1, 1),
      "1024x1024 with min=1,max=2 should pick (1,1)"
    );
    assert_eq!(
      g.thumbnail(),
      None,
      "1×1 selection in multi-tile branch must NOT attach a thumbnail"
    );
  }

  #[test]
  fn min_max_tiles_one_routes_single_tile_even_for_large_image() {
    // Upstream resize_and_split sets do_image_splitting = !(min_tiles==max_tiles==1)
    // and only enters the multi-tile branch when (is_image_large && do_image_splitting).
    // Without the gate, a 4096x2160 image with this budget would fall into the
    // multi-tile branch, get rows=cols=1 (only candidate), and attach a thumbnail —
    // yielding a TileGrid that build_image_block treats as single-tile (no thumbnail
    // marker emitted) while PreprocessedImage's tile count includes the thumbnail.
    let mut budget = ImageBudget::new();
    budget.set_min_tiles(1);
    budget.set_max_tiles(1);
    let g = pick_tile_grid(4096, 2160, &budget).expect("budget is valid");
    assert_eq!(
      (g.rows(), g.cols()),
      (1, 1),
      "min=max=1 must route to single-tile branch"
    );
    assert_eq!(
      g.thumbnail(),
      None,
      "single-tile branch must never attach a thumbnail"
    );
    // Tile dims come from smart_resize (not 512x512), so they'll be one of the
    // multiples-of-32 grid steps shrunk to fit max_image_tokens.
    assert!(g.tile_h().is_multiple_of(TILE_PIXEL_UNIT));
    assert!(g.tile_w().is_multiple_of(TILE_PIXEL_UNIT));
  }

  /// N8 parity fixtures: 28 cases covering all four budget-envelope corners,
  /// single-tile path (below threshold), multi-tile path (above threshold),
  /// aspect-ratio tie-break pairs, thumbnail boundary, pathological aspects,
  /// and non-default budgets.
  ///
  /// Expected values derived from the upstream Python via
  /// `scripts/derive_tile_grid_fixtures.py` which calls
  /// `Lfm2VlImageProcessorFast._is_image_too_large`, `.smart_resize`, and
  /// `find_closest_aspect_ratio` directly.
  ///
  /// Rows/cols semantic: upstream `find_closest_aspect_ratio` returns
  /// `(grid_width, grid_height)` which the upstream caller stores as
  /// `(num_rows, num_cols)` respectively (naming inversion in upstream).
  /// Here `rows` = height-direction tile count = upstream `grid_height`,
  /// and `cols` = width-direction tile count = upstream `grid_width`.
  #[test]
  fn pick_tile_grid_parity_cases() {
    // (src_w, src_h, budget, expected_TileGrid)
    // TileGrid::new(rows, cols, tile_h, tile_w, thumbnail: Option<(thumb_h, thumb_w)>)
    let cases: &[(u32, u32, ImageBudget, TileGrid)] = &[
      // ── Single-tile path ────────────────────────────────────────────────────
      //
      // small_square_256: area=65536, within budget; no scaling
      (
        256,
        256,
        ImageBudget::new(),
        TileGrid::new(1, 1, 256, 256, None),
      ),
      //
      // small_square_512: area=262144 = max_area exactly; no scaling
      (
        512,
        512,
        ImageBudget::new(),
        TileGrid::new(1, 1, 512, 512, None),
      ),
      //
      // just_below_threshold_723x724: _is_image_too_large rounds 723→736, 724→736,
      //   product=736*736=541696 > threshold=524288 → MULTI-TILE (Fix 2 changes this case)
      //   aspect=723/724≈0.999 → 2×2; thumbnail: smart_resize(723,724)→(480,512)
      (
        723,
        724,
        ImageBudget::new(),
        TileGrid::new(2, 2, 512, 512, Some((512, 480))),
      ),
      //
      // tiny_32x32: area=1024 << min_area; grow path; 32×32→256×256
      (
        32,
        32,
        ImageBudget::new(),
        TileGrid::new(1, 1, 256, 256, None),
      ),
      //
      // 4:3 within budget: 320×240; h_bar=max(32,round(240/32)*32)=256, w_bar=320
      (
        320,
        240,
        ImageBudget::new(),
        TileGrid::new(1, 1, 256, 320, None),
      ),
      //
      // 16:9 within budget: 384×216; h_bar=max(32,round(216/32)*32)=224, w_bar=384
      (
        384,
        216,
        ImageBudget::new(),
        TileGrid::new(1, 1, 224, 384, None),
      ),
      //
      // 4:3 above max_area: 640×480; shrink path; β=sqrt(640*480/262144)≈1.0801;
      //   h_bar=floor(480/β/32)*32=416, w_bar=floor(640/β/32)*32=576  (Fix 3: was 448)
      (
        640,
        480,
        ImageBudget::new(),
        TileGrid::new(1, 1, 416, 576, None),
      ),
      //
      // 16:9 below threshold: 480×270; h_bar=max(32,round(270/32)*32)=256, w_bar=480
      (
        480,
        270,
        ImageBudget::new(),
        TileGrid::new(1, 1, 256, 480, None),
      ),
      //
      // ── Pathological aspects (single-tile) ──────────────────────────────────
      //
      // 32×1024: grow path; β=sqrt(65536/(32*1024))=sqrt(2)≈1.4142;
      //   h_bar=ceil(1024*β/32)*32=1472, w_bar=ceil(32*β/32)*32=64  (Fix 3: was 1440,32)
      (
        32,
        1024,
        ImageBudget::new(),
        TileGrid::new(1, 1, 1472, 64, None),
      ),
      //
      // 1024×32: transposed; tile_h=64, tile_w=1472
      (
        1024,
        32,
        ImageBudget::new(),
        TileGrid::new(1, 1, 64, 1472, None),
      ),
      //
      // 1×8000: h_bar=max(32,round_by_factor(8000,32))=8000; w_bar=max(32,0)=32;
      //   product=256000 within [65536,262144] → no scaling; (Fix 3: was 22912,32)
      (
        1,
        8000,
        ImageBudget::new(),
        TileGrid::new(1, 1, 8000, 32, None),
      ),
      //
      // 8000×1: transposed; tile_h=32, tile_w=8000
      (
        8000,
        1,
        ImageBudget::new(),
        TileGrid::new(1, 1, 32, 8000, None),
      ),
      //
      // ── Multi-tile path ──────────────────────────────────────────────────────
      //
      // 1024×1024: aspect=1.0 → 2×2; thumbnail: smart_resize→(512,512)
      (
        1024,
        1024,
        ImageBudget::new(),
        TileGrid::new(2, 2, 512, 512, Some((512, 512))),
      ),
      //
      // 768×768: aspect=1.0 → 2×2; thumbnail: smart_resize→(512,512)
      (
        768,
        768,
        ImageBudget::new(),
        TileGrid::new(2, 2, 512, 512, Some((512, 512))),
      ),
      //
      // 1920×1080: aspect≈1.778 → rows=2, cols=4; thumbnail: smart_resize→(384,672)
      (
        1920,
        1080,
        ImageBudget::new(),
        TileGrid::new(2, 4, 512, 512, Some((384, 672))),
      ),
      //
      // 1080×1920: aspect≈0.5625 → rows=4, cols=2; thumbnail: smart_resize→(672,384)
      (
        1080,
        1920,
        ImageBudget::new(),
        TileGrid::new(4, 2, 512, 512, Some((672, 384))),
      ),
      //
      // 1280×720: aspect≈1.778 → rows=1, cols=2; thumbnail→(384,672)
      (
        1280,
        720,
        ImageBudget::new(),
        TileGrid::new(1, 2, 512, 512, Some((384, 672))),
      ),
      //
      // 2560×1440: same aspect as 1920×1080 → rows=2, cols=4; thumbnail same
      (
        2560,
        1440,
        ImageBudget::new(),
        TileGrid::new(2, 4, 512, 512, Some((384, 672))),
      ),
      //
      // 1440×2560: 4×2; thumbnail same
      (
        1440,
        2560,
        ImageBudget::new(),
        TileGrid::new(4, 2, 512, 512, Some((672, 384))),
      ),
      //
      // ── Aspect-ratio tie-break pairs ─────────────────────────────────────────
      //
      // 1024×768: aspect≈1.333 → rows=2, cols=3;
      //   thumbnail: smart_resize(1024,768)→(576,416)  (Fix 3: was 576,448)
      (
        1024,
        768,
        ImageBudget::new(),
        TileGrid::new(2, 3, 512, 512, Some((416, 576))),
      ),
      //
      // 768×1024: aspect≈0.75 → rows=3, cols=2; thumbnail→(576,416)  (Fix 3: was 576,448)
      (
        768,
        1024,
        ImageBudget::new(),
        TileGrid::new(3, 2, 512, 512, Some((576, 416))),
      ),
      //
      // ── 2×4 vs 1×2 boundary (Fix 4: was 1×2, now 2×4) ──────────────────────
      //
      // 1600×800: aspect=2.0; candidates with diff=0: (2,1),(4,2),(6,3),(8,4),(10,5);
      //   sorted by product; first (2,1): new best. Then (4,2): tie, area=1280000 >
      //   0.5*2097152=1048576 → override → rows=2, cols=4.
      //   thumbnail: smart_resize(1600,800)→(352,704)  (Fix 3+4 combined)
      (
        1600,
        800,
        ImageBudget::new(),
        TileGrid::new(2, 4, 512, 512, Some((352, 704))),
      ),
      //
      // 800×1600: aspect=0.5 → rows=4, cols=2; thumbnail→(704,352)
      (
        800,
        1600,
        ImageBudget::new(),
        TileGrid::new(4, 2, 512, 512, Some((704, 352))),
      ),
      //
      // ── Just-above/below threshold boundary ──────────────────────────────────
      //
      // 720×730: _is_image_too_large: h_bar=max(16,round(730/32)*32)=736,
      //   w_bar=max(16,round(720/32)*32)=max(16,22*32)=704 (banker's round(22.5)=22),
      //   product=736*704=518144 < 524288 → SINGLE-TILE  (Fix 2: was multi-tile 2×2)
      //   smart_resize: h_bar=736, w_bar=704, product=518144 < 65536? No. > 262144? No.
      //   So no scaling: new_h=736 but wait...
      //   Actually smart_resize starts with round_by_factor then checks rounded area.
      //   h_bar=max(32,round_by_factor(730,32))=max(32,736)=736
      //   w_bar=max(32,round_by_factor(720,32))=max(32,704)=704
      //   rounded_area=736*704=518144 in [65536,262144]? No: 518144 > 262144.
      //   So shrink: beta=sqrt(720*730/262144)=sqrt(525600/262144)≈1.4163
      //   h_bar=max(32,floor(730/β/32)*32)=max(32,floor(16.094)*32)=max(32,512)=512
      //   w_bar=max(32,floor(720/β/32)*32)=max(32,floor(15.875)*32)=max(32,480)=480
      //   → (new_w=480, new_h=512)
      (
        720,
        730,
        ImageBudget::new(),
        TileGrid::new(1, 1, 512, 480, None),
      ),
      //
      // ── Non-default budgets ───────────────────────────────────────────────────
      //
      // fast budget: max_image_tokens=64, max_tiles=4, no thumbnail
      // threshold=64*256*4*2.0=131072; 256²=65536 < 131072 → single
      (
        256,
        256,
        ImageBudget::new()
          .with_min_image_tokens(32)
          .with_max_image_tokens(64)
          .with_min_tiles(2)
          .with_max_tiles(4)
          .with_use_thumbnail(false),
        TileGrid::new(1, 1, 256, 256, None),
      ),
      //
      // fast budget: 1024×1024 → multi (area=1048576 > threshold=131072), no thumbnail
      (
        1024,
        1024,
        ImageBudget::new()
          .with_min_image_tokens(32)
          .with_max_image_tokens(64)
          .with_min_tiles(2)
          .with_max_tiles(4)
          .with_use_thumbnail(false),
        TileGrid::new(2, 2, 512, 512, None),
      ),
      //
      // default budget, min_tiles=4: 1920×1080 → rows=2, cols=4 (8 tiles ≥ 4)
      (
        1920,
        1080,
        ImageBudget::new().with_min_tiles(4).with_max_tiles(10),
        TileGrid::new(2, 4, 512, 512, Some((384, 672))),
      ),
      //
      // default budget, min_tiles=4: 1080×1920 → rows=4, cols=2
      (
        1080,
        1920,
        ImageBudget::new().with_min_tiles(4).with_max_tiles(10),
        TileGrid::new(4, 2, 512, 512, Some((672, 384))),
      ),
    ];

    for (w, h, budget, expected) in cases {
      let actual = pick_tile_grid(*w, *h, budget).unwrap();
      assert_eq!(actual, *expected, "case w={w} h={h}");
    }
  }
}
