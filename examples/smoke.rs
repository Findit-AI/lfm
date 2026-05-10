//! Free-form chat with one image.
//!
//! Usage (standard layout — tokenizer.json in model dir):
//! ```bash
//! LFM_MODEL_PATH=/path/to/LFM2.5-VL-450M-ONNX \
//!   cargo run --example smoke --features inference,decoders -- /path/to/image.jpg "Describe this."
//! ```
//!
//! Usage (ONNX-only dir — tokenizer + configs are bundled into the crate):
//! ```bash
//! LFM_ONNX_PATH=/path/with/onnx-files-only \
//!   cargo run --example smoke --features bundled,decoders -- /path/to/image.jpg "Describe this."
//! ```

#[cfg(feature = "inference")]
fn main() -> lfm::Result<()> {
  let mut args = std::env::args().skip(1);
  let image_path = args.next().expect("usage: smoke <image> <prompt>");
  let prompt = args.next().unwrap_or_else(|| "Describe this image.".into());

  let mut engine = if let Ok(onnx_dir) = std::env::var("LFM_ONNX_PATH") {
    // Bundled tokenizer + configs path: supply only the ONNX directory.
    #[cfg(feature = "bundled")]
    {
      lfm::Engine::from_onnx_dir(onnx_dir, lfm::Options::default())?
    }
    #[cfg(not(feature = "bundled"))]
    {
      let _ = onnx_dir;
      panic!(
        "LFM_ONNX_PATH set but the `bundled` feature is not enabled; \
              rebuild with --features bundled or use LFM_MODEL_PATH instead"
      );
    }
  } else {
    let model_dir = std::env::var("LFM_MODEL_PATH").expect(
      "set LFM_MODEL_PATH=/path/to/LFM2.5-VL-450M-ONNX \
       or LFM_ONNX_PATH=/path/with/onnx-only (requires --features bundled)",
    );
    lfm::Engine::from_dir(&model_dir, lfm::Options::default())?
  };

  let messages = vec![lfm::ChatMessage::new(
    smol_str::SmolStr::new_static("user"),
    lfm::ChatContent::Parts(vec![
      lfm::ContentPart::Image,
      lfm::ContentPart::Text(prompt),
    ]),
  )];
  let images = vec![lfm::ImageInput::Path(std::path::Path::new(&image_path))];

  let text = engine.generate(&messages, &images, &lfm::RequestOptions::default())?;
  println!("{text}");
  Ok(())
}

#[cfg(not(feature = "inference"))]
fn main() {
  eprintln!("smoke requires the `inference` feature");
}
