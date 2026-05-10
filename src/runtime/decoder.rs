//! Decoder session wrapper + hybrid KV/conv cache management.
//!
//! Per spec §8.1:
//! - 10 conv layers at sparse indices [0,1,3,4,6,7,9,11,13,15]
//!   with FIXED-shape cache [1, 1024, 3] zero-filled at step 0.
//! - 6 attn layers at sparse indices [2,5,8,10,12,14] with
//!   DYNAMIC-shape KV cache [1, 8, past_len, 64], past_len=0 at step 0.
//! - Decoder has NO position_ids input (G1).

use std::{collections::HashMap, path::Path};

use ort::{
  memory::Allocator,
  session::Session,
  value::{Tensor, TensorRef},
};
use smol_str::SmolStr;

use crate::{
  error::{Error, Result},
  options::Options,
  runtime::session::{build_session, collect_cache_inputs, validate_decoder_session},
};

/// Wraps `decoder_model_merged.onnx`. Owns one `ort::Session`
/// + the cache template (names + shapes discovered at session-build time).
#[allow(dead_code)]
pub(crate) struct Decoder {
  session: Session,
  template: KvCacheTemplate,
}

/// Names + initial shapes for cache tensors, discovered from the
/// decoder session's input metadata. Used by `Decoder::new_cache` to
/// zero-init a fresh `KvCache` and by `advance` to map present_X → past_X.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct KvCacheTemplate {
  /// past_conv.{i} names paired with their shapes (always [1, 1024, 3]).
  /// Stored as Vec<i64> for consistency with attn shapes.
  conv: Vec<(SmolStr, Vec<i64>)>,
  /// past_key_values.{i}.{key,value} names paired with [1, 8, -1, 64]
  /// where -1 marks the dynamic past_len axis.
  attn: Vec<(SmolStr, Vec<i64>)>,
  /// Map present_X (output name) → past_X (input name).
  present_to_past: HashMap<SmolStr, SmolStr>,
}

/// Per-call hybrid cache. Holds owned `Tensor<f32>` per cache slot,
/// keyed by the past-input name (e.g. "past_conv.0", "past_key_values.2.key").
/// Both maps grow as `Decoder::step` writes present-outputs back.
#[allow(dead_code)]
pub(crate) struct KvCache {
  conv: HashMap<SmolStr, Tensor<f32>>,
  attn: HashMap<SmolStr, Tensor<f32>>,
  /// Total tokens fed so far (S at step 0; S + step_count after each step).
  pub(crate) past_len: usize,
}

impl Decoder {
  #[allow(dead_code)]
  pub(crate) fn from_path(path: &Path, opts: &Options) -> Result<Self> {
    let session = build_session(path, opts)?;
    validate_decoder_session(&session)?;
    let template = build_template(&session)?;
    Ok(Self { session, template })
  }

  #[allow(dead_code)]
  pub(crate) fn from_session(session: Session) -> Result<Self> {
    validate_decoder_session(&session)?;
    let template = build_template(&session)?;
    Ok(Self { session, template })
  }

  /// Construct a fresh KvCache:
  /// - past_conv.{i}: zero-filled at FIXED shape [1, 1024, 3]
  /// - past_key_values.{i}.{key,value}: empty at [1, 8, 0, 64]
  ///
  /// `Tensor::from_array` rejects any zero-dim shape with
  /// `Invalid dimension #N; all dimensions must be >= 1 when
  /// creating a tensor from raw data` (ort 2.0 hard check), which
  /// makes it unusable for the empty `past_len = 0` initialization
  /// of the attn cache. `Tensor::new(allocator, shape)` allocates
  /// directly via ONNX Runtime and accepts shapes with zero dims,
  /// returning an uninitialized buffer of the requested layout.
  #[allow(dead_code)]
  pub(crate) fn new_cache(&self) -> Result<KvCache> {
    let alloc = Allocator::default();
    let mut conv = HashMap::with_capacity(self.template.conv.len());
    for (name, shape_i64) in &self.template.conv {
      // Convert i64 → usize directly (no -1 in conv shapes).
      let shape: Vec<usize> = shape_i64.iter().map(|&d| d as usize).collect();
      let total: usize = shape.iter().product();
      // Conv cache is non-empty (1*1024*3 = 3072 elements) and
      // needs to start zeroed, so build it from a zero Vec via
      // `from_array`.
      let tensor = Tensor::from_array((shape.as_slice(), vec![0f32; total])).map_err(Error::Ort)?;
      conv.insert(name.clone(), tensor);
    }
    let mut attn = HashMap::with_capacity(self.template.attn.len());
    for (name, shape_i64) in &self.template.attn {
      // Resolve -1 → 0 (empty initialization for past_len axis).
      let shape: Vec<usize> = shape_i64
        .iter()
        .map(|&d| if d < 0 { 0 } else { d as usize })
        .collect();
      // `Allocator::new` accepts zero-dim shapes; the resulting
      // tensor has `num_elements() == 0`. Decoder reads past_len
      // from `cache.past_len` (initialized to 0 below), not from
      // these tensors' shape, so the zero-element buffer is fine.
      let tensor = Tensor::<f32>::new(&alloc, shape.as_slice()).map_err(Error::Ort)?;
      attn.insert(name.clone(), tensor);
    }
    Ok(KvCache {
      conv,
      attn,
      past_len: 0,
    })
  }

  /// Run one decoder step. Returns flat logits from the LAST sequence
  /// position: `[vocab_size = 65536]`.
  ///
  /// `inputs_embeds` is `[1, seq_len, 1024]` flattened.
  /// `seq_len` is the number of NEW tokens this step.
  /// Mutates `cache` in place (advances past_len + swaps present_* → past_*).
  #[allow(dead_code)]
  pub(crate) fn step(
    &mut self,
    cache: &mut KvCache,
    inputs_embeds: &[f32],
    seq_len: usize,
  ) -> Result<Vec<f32>> {
    let total_len = cache.past_len + seq_len;
    let attn_mask: Vec<i64> = vec![1i64; total_len];

    let inputs_shape = [1usize, seq_len, 1024usize];
    let mask_shape = [1usize, total_len];

    let embeds_ref =
      TensorRef::from_array_view((inputs_shape, inputs_embeds)).map_err(Error::Ort)?;
    let mask_ref =
      TensorRef::from_array_view((mask_shape, attn_mask.as_slice())).map_err(Error::Ort)?;

    // Build inputs using the incremental construction pattern from ort docs.
    // ort::inputs!["k" => v] returns Vec<(Cow<str>, SessionInputValue<'_>)>.
    // We then push cache tensors via &Tensor<f32> → SessionInputValue::View.
    let mut my_inputs = ort::inputs![
        "inputs_embeds" => embeds_ref,
        "attention_mask" => mask_ref,
    ];

    for (name, tensor) in cache.conv.iter().chain(cache.attn.iter()) {
      // &Tensor<f32> implements Into<SessionInputValue<'_>> via the
      // From<&'v Value<T>> impl (calls value.view().into_dyn()).
      my_inputs.push((name.as_str().into(), tensor.into()));
    }

    let outputs = self.session.run(my_inputs).map_err(Error::Ort)?;

    // Extract logits — last position only.
    let logits_out = outputs.get("logits").ok_or(Error::SessionShapeMismatch {
      input: "logits",
      expected: "output present in session run",
      got: vec![],
    })?;
    let (shape, data) = logits_out.try_extract_tensor::<f32>().map_err(Error::Ort)?;
    if shape.len() != 3 || shape[0] < 1 || shape[1] < 1 || shape[2] != 65536 {
      return Err(Error::SessionShapeMismatch {
        input: "logits",
        expected: "[batch>=1, seq>=1, 65536]",
        got: shape.to_vec(),
      });
    }
    let last_pos = (shape[1] - 1) as usize;
    let vocab = shape[2] as usize;
    let logits = data[last_pos * vocab..(last_pos + 1) * vocab].to_vec();

    // Advance the cache: present_X → past_X.
    advance_cache(cache, &outputs, &self.template.present_to_past)?;
    cache.past_len = total_len;
    Ok(logits)
  }
}

/// Build cache template from session metadata (called once at construction).
///
/// Validates that every `past_*` cache input has a corresponding
/// `present_*` output. Without this check a future ONNX revision that
/// renamed or dropped a `present_*` output would still pass session
/// construction; `advance_cache` would then silently leave the cache
/// stale, and generation would corrupt without a clear error.
fn build_template(session: &Session) -> Result<KvCacheTemplate> {
  let inputs = collect_cache_inputs(session.inputs())?;
  let mut conv = Vec::with_capacity(inputs.conv.len());
  for name in inputs.conv {
    // Static [1, 1024, 3] for all conv layers.
    conv.push((SmolStr::from(name), vec![1i64, 1024, 3]));
  }
  let mut attn = Vec::with_capacity(inputs.attn.len());
  for name in inputs.attn {
    // [1, 8, -1, 64] template; -1 marks dynamic past_len.
    attn.push((SmolStr::from(name), vec![1i64, 8, -1, 64]));
  }
  let present_to_past = build_present_to_past(
    &session
      .outputs()
      .iter()
      .map(|o| SmolStr::from(o.name()))
      .collect::<Vec<_>>(),
  );

  // Bijection check: every past_* cache input must be reachable from
  // some present_* output via the name map.
  let mapped_pasts: std::collections::HashSet<&SmolStr> = present_to_past.values().collect();
  let mut missing: Vec<i64> = Vec::new();
  for (past_name, _) in conv.iter().chain(attn.iter()) {
    if !mapped_pasts.contains(past_name) {
      // Reuse SessionShapeMismatch.got as a "missing names" channel by
      // emitting a hash placeholder; the message string is the actual signal.
      missing.push(past_name.len() as i64);
    }
  }
  if !missing.is_empty() {
    return Err(Error::SessionShapeMismatch {
      input: "present_*",
      expected: "one present_* output per past_* cache input",
      got: missing,
    });
  }

  Ok(KvCacheTemplate {
    conv,
    attn,
    present_to_past,
  })
}

/// Map present_* output names → past_* input names.
fn build_present_to_past(present_names: &[SmolStr]) -> HashMap<SmolStr, SmolStr> {
  let mut map = HashMap::new();
  for n in present_names {
    if let Some(rest) = n.strip_prefix("present_conv.") {
      map.insert(n.clone(), SmolStr::from(format!("past_conv.{rest}")));
    } else if let Some(rest) = n.strip_prefix("present.") {
      map.insert(n.clone(), SmolStr::from(format!("past_key_values.{rest}")));
    }
  }
  map
}

/// Walk session outputs, extract present-* tensors, store under past-*
/// keys for the next step. Splits between conv and attn maps based on
/// the past-name prefix. Errors on any missing present output — the
/// bijection was verified at construction, so a missing tensor at this
/// point would mean a per-step inconsistency that would silently
/// freeze the cache for that layer if we continued.
fn advance_cache(
  cache: &mut KvCache,
  outputs: &ort::session::SessionOutputs<'_>,
  present_to_past: &HashMap<SmolStr, SmolStr>,
) -> Result<()> {
  for (present_name, past_name) in present_to_past {
    let Some(out) = outputs.get(present_name.as_str()) else {
      return Err(Error::SessionShapeMismatch {
        input: "present_*",
        expected: "every mapped present_* output present in session.run() result",
        got: vec![present_name.len() as i64],
      });
    };
    let (shape, data) = out.try_extract_tensor::<f32>().map_err(Error::Ort)?;
    let shape_usize: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let new_tensor =
      Tensor::from_array((shape_usize.as_slice(), data.to_vec())).map_err(Error::Ort)?;
    if past_name.starts_with("past_conv.") {
      cache.conv.insert(past_name.clone(), new_tensor);
    } else {
      cache.attn.insert(past_name.clone(), new_tensor);
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn build_present_to_past_maps_conv_and_attn() {
    let names: Vec<SmolStr> = vec![
      "present_conv.0".into(),
      "present_conv.13".into(),
      "present.2.key".into(),
      "present.14.value".into(),
      "logits".into(), // not a cache output; should be skipped
    ];
    let map = build_present_to_past(&names);
    assert_eq!(
      map
        .get(&SmolStr::from("present_conv.0"))
        .map(SmolStr::as_str),
      Some("past_conv.0")
    );
    assert_eq!(
      map
        .get(&SmolStr::from("present_conv.13"))
        .map(SmolStr::as_str),
      Some("past_conv.13")
    );
    assert_eq!(
      map
        .get(&SmolStr::from("present.2.key"))
        .map(SmolStr::as_str),
      Some("past_key_values.2.key")
    );
    assert_eq!(
      map
        .get(&SmolStr::from("present.14.value"))
        .map(SmolStr::as_str),
      Some("past_key_values.14.value")
    );
    assert_eq!(map.get(&SmolStr::from("logits")), None);
  }
}
