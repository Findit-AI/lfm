//! Demonstrate the wasm-friendly preprocessing subset (no inference needed).
//!
//! This example compiles without any inference features and exercises the
//! public `Preprocessor` + `ImageBudget` API directly.
//!
//! Usage:
//! ```bash
//! cargo run --example preprocess_only --no-default-features --features decoders \
//!   -- /path/to/image.jpg
//! ```

#[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
  let image_path = std::env::args()
    .nth(1)
    .expect("usage: preprocess_only <image>");

  let img = lfm::decode_with_orientation(std::path::Path::new(&image_path))?;
  let budget = lfm::ImageBudget::default();
  let preproc = lfm::Preprocessor::new(budget);
  let result = preproc.preprocess(&img)?;

  println!("Image:          {image_path}");
  println!(
    "Tiles:          {}×{} ({})",
    result.cols(),
    result.rows(),
    result.num_tiles()
  );
  println!("Tile size:      {:?} px", result.main_tile_size());
  println!(
    "Thumbnail:      {}",
    result
      .thumbnail_size()
      .map_or_else(|| "none".into(), |s| format!("{s:?} px"))
  );
  println!("Image tokens:   {}", result.num_image_tokens());
  println!("pixel_values:   {} f32 values", result.pixel_values().len());
  Ok(())
}

// Codex round 37 finding 1: under --no-default-features, the
// decoder-using main() is cfg'd out. Provide a stub so the example
// still builds and clippy doesn't trip on unreachable / unused
// bindings inside a single conditional main().
#[cfg(not(all(feature = "decoders", not(target_arch = "wasm32"))))]
fn main() {
  let _ = std::env::args().nth(1);
  println!(
    "preprocess_only: enable the `decoders` feature on a non-wasm target to run this example."
  );
}
