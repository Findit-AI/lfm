//! EmbedTokens — wraps `embed_tokens.onnx`.

use std::path::Path;

use ort::{session::Session, value::TensorRef};

use crate::{
  error::{Error, Result},
  options::Options,
  runtime::session::{build_session, validate_embed_session},
};

/// Wraps `embed_tokens.onnx`. Owns one `ort::Session`.
#[allow(dead_code)]
pub(crate) struct EmbedTokens {
  session: Session,
}

impl EmbedTokens {
  #[allow(dead_code)]
  pub(crate) fn from_path(path: &Path, opts: &Options) -> Result<Self> {
    let session = build_session(path, opts)?;
    validate_embed_session(&session)?;
    Ok(Self { session })
  }

  #[allow(dead_code)]
  pub(crate) fn from_session(session: Session) -> Result<Self> {
    validate_embed_session(&session)?;
    Ok(Self { session })
  }

  /// Embed a sequence of token IDs. Returns flat `[seq_len * 1024]`.
  #[allow(dead_code)]
  pub(crate) fn run(&mut self, input_ids: &[i64]) -> Result<Vec<f32>> {
    let shape = [1usize, input_ids.len()];
    let ids = TensorRef::from_array_view((shape, input_ids)).map_err(Error::Ort)?;
    let outputs = self
      .session
      .run(ort::inputs!["input_ids" => ids])
      .map_err(Error::Ort)?;
    let out = outputs
      .get("inputs_embeds")
      .ok_or(Error::SessionShapeMismatch {
        input: "inputs_embeds",
        expected: "output present in session run",
        got: vec![],
      })?;
    let (s, data) = out.try_extract_tensor::<f32>().map_err(Error::Ort)?;
    // validate the FULL shape, not just
    // rank-and-last-dim. A drifted embed_tokens.onnx whose metadata
    // still matches `[-1, -1, 1024]` could return fewer sequence
    // positions; the caller's `debug_assert_eq!` would not catch
    // that in release builds, and image-splice indexing would walk
    // off the embedding buffer — silent panic or opaque decoder
    // failure instead of a recoverable SessionShapeMismatch.
    let expected_seq = input_ids.len() as i64;
    if s.len() != 3 || s[0] != 1 || s[1] != expected_seq || s[2] != 1024 {
      return Err(Error::SessionShapeMismatch {
        input: "inputs_embeds",
        expected: "[1, input_ids.len(), 1024]",
        got: s.to_vec(),
      });
    }
    let expected_len = input_ids.len().saturating_mul(1024);
    if data.len() != expected_len {
      return Err(Error::SessionShapeMismatch {
        input: "inputs_embeds",
        expected: "buffer length input_ids.len() * 1024",
        got: vec![data.len() as i64],
      });
    }
    Ok(data.to_vec())
  }
}
