//! VisionEncoder — wraps `vision_encoder.onnx`.
//! Single-image only; multi-image callers MUST loop (see spec §7.5).

use std::path::Path;

use ort::{session::Session, value::TensorRef};

use crate::{
  error::{Error, Result},
  options::Options,
  preproc::PreprocessedImage,
  runtime::session::{build_session, validate_vision_session},
};

/// Wraps `vision_encoder.onnx`. Owns one `ort::Session`.
///
/// **Single-image only** (per spec §7.5): batched calls across
/// multiple images silently corrupt outputs when any image
/// routes through the multi-tile path. `Engine::run`/`generate` iterate
/// per-image and concatenate the flat `image_features` outputs in
/// source order.
#[allow(dead_code)]
pub(crate) struct VisionEncoder {
  session: Session,
}

impl VisionEncoder {
  /// Construct from an ONNX file path. Validates outlet contract at build.
  #[allow(dead_code)]
  pub(crate) fn from_path(path: &Path, opts: &Options) -> Result<Self> {
    let session = build_session(path, opts)?;
    validate_vision_session(&session)?;
    Ok(Self { session })
  }

  /// Construct from a caller-built `Session`. Validates outlet contract.
  #[allow(dead_code)]
  pub(crate) fn from_session(session: Session) -> Result<Self> {
    validate_vision_session(&session)?;
    Ok(Self { session })
  }

  /// Run vision encoding on one preprocessed image. Returns flat
  /// `[num_image_tokens * 1024]` `image_features`.
  ///
  /// **Single-image only** — see struct doc for the multi-image contract.
  #[allow(dead_code)]
  pub(crate) fn run(&mut self, img: &PreprocessedImage) -> Result<Vec<f32>> {
    let n_batch = img.batch_size();
    let num_patches = img.patches_per_entry();
    let pv_shape = [n_batch, num_patches, 768usize];
    let mask_shape = [n_batch, num_patches];
    let sp_shape = [n_batch, 2usize];

    let pv = TensorRef::from_array_view((pv_shape, img.pixel_values())).map_err(Error::Ort)?;
    let mask =
      TensorRef::from_array_view((mask_shape, img.pixel_attention_mask())).map_err(Error::Ort)?;
    let sp = TensorRef::from_array_view((sp_shape, img.spatial_shapes())).map_err(Error::Ort)?;

    let outputs = self
      .session
      .run(ort::inputs![
        "pixel_values"         => pv,
        "pixel_attention_mask" => mask,
        "spatial_shapes"       => sp,
      ])
      .map_err(Error::Ort)?;

    let out = outputs
      .get("image_features")
      .ok_or(Error::SessionShapeMismatch {
        input: "image_features",
        expected: "output present in session run",
        got: vec![],
      })?;
    let (shape, data) = out.try_extract_tensor::<f32>().map_err(Error::Ort)?;
    if shape.len() != 2 {
      return Err(Error::SessionShapeMismatch {
        input: "image_features",
        expected: "rank 2",
        got: shape.to_vec(),
      });
    }
    if shape[1] != 1024 {
      return Err(Error::SessionShapeMismatch {
        input: "image_features",
        expected: "second dim = 1024",
        got: shape.to_vec(),
      });
    }
    // Reject NaN/Inf at the session boundary. Vision embeddings get
    // spliced into the text-embedding stream; a single NaN would
    // poison the decoder's attention for the entire generation.
    if data.iter().any(|v| !v.is_finite()) {
      return Err(Error::SessionNonFiniteOutput {
        stage: "vision_encoder",
      });
    }
    Ok(data.to_vec())
  }
}
