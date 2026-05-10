//! The image-analysis preset: [`ImageAnalysisTask`] produces a
//! typed [`ImageAnalysis`] with nine fields (scene category,
//! free-form description, five detection lists, shot-type label,
//! search tags). The "scene" wording survives in the `scene` field
//! and in the prompt because the upstream use case is video keyframes
//! representing scenes — but the type itself is engine-output for a
//! single image and works for any single-image analysis pipeline.
//!
//! `ImageAnalysis` lives in the `llmtask` sibling crate (re-exported
//! at the top of this module); this engine doesn't depend on
//! `findit-proto`, so any consumer can map the result into its own
//! wire shape. The legacy `findit-proto::database::SceneVlmResult`
//! paired each detection-array entry with a `confidence` float;
//! `llmtask::ImageAnalysis` exposes those buckets as plain
//! `Vec<SmolStr>`. VLM self-reported per-detection confidence is
//! poorly calibrated, and a flat hardcoded confidence on every entry
//! is a no-op for both UX and search-time ranking. If a downstream
//! consumer needs per-detection scores, the right place to get them
//! is from search-time embedding similarity or scene-aggregation
//! metrics (frame frequency, etc.), not from VLM self-report. The
//! `findit-proto` mapping (when revived) can stamp a fixed value on
//! its side or compute one from those non-VLM sources.
//!
//! `colors` is intentionally NOT a VLM output: dominant-color
//! extraction is a closed-form image-processing problem (k-means /
//! histogram clustering on pixel data + a perceptual-distance lookup
//! against a named-color dataset like xkcd's), so making the VLM emit
//! it would be slower, less accurate, and non-deterministic compared
//! to running the algorithm on the keyframes directly. That belongs
//! in whatever orchestrates keyframes → final record, not in this
//! crate. `lighting` stays — semantic lighting terms ("backlit",
//! "spotlight", "golden hour") need scene-level visual reasoning that
//! pixel statistics alone can't reproduce.

use serde::Deserialize;
use serde_json::{Value, json};
use smol_str::SmolStr;

use llmtask::{JsonParseError, Task};

pub use llmtask::ImageAnalysis;

/// The scene-analysis prompt — verbatim port from `qwen/src/scene.rs`.
// IMAGE_ANALYSIS_PROMPT is intentionally written WITHOUT enumerated example
// values. in deterministic
// (greedy) mode, mistralrs 0.8's `presence_penalty` is applied over
// `seq.get_toks()` (prompt + generated tokens), so every value-token
// the prompt enumerates as an example gets a `-presence_penalty`
// logit shift before the model emits anything. Listing example values
// like "office", "wide shot", or "birthday cake with candles"
// systematically biases the model AWAY from those exact terms when a
// scene legitimately matches one. The fix here removes value-token
// examples from the prompt; format guidance moves to descriptive
// constraints (word counts, lowercase) so the model still knows the
// expected shape without enumerating the vocabulary it's penalized
// against.
const IMAGE_ANALYSIS_PROMPT: &str = r#"Analyze the following video keyframes (in chronological order) from a single scene.

Return ONLY a valid JSON object with exactly these fields:
scene: a single short scene-category label in lowercase English, 1-3 words, no full sentence.
description: 1-2 concise sentences in English describing the stable visual facts across the scene. Cover who is present, what they are doing, the setting, and the overall mood or visual style. If readable on-screen text appears, quote that text first, then continue the description.
subjects: array of distinct people or animals as short noun phrases (each 2-6 words) with visible distinguishing features.
objects: array of notable, search-relevant objects as short noun phrases (each 2-6 words).
actions: array of visible actions as short verb phrases (each 1-4 words).
mood: array of single-word or two-word adjectives describing the scene's overall emotional tone.
shot_type: a single short camera-shot label in lowercase English, 1-2 words (a cinematography term).
lighting: array of single-word or two-word lighting descriptors.
tags: array of 8-12 short English search tags in lowercase. Prefer high-confidence search terms, complementary synonyms, style words, and culture-specific terms only when visually supported.

Rules:
- Use only information supported by the keyframes.
- Prefer concrete visual facts over speculation.
- Keep arrays deduplicated.
- Use empty arrays or empty strings when a field is unknown.
- Do not return markdown or any text outside the JSON object."#;

const REQUIRED_FIELDS: &[&str] = &[
  "scene",
  "description",
  "subjects",
  "objects",
  "actions",
  "mood",
  "shot_type",
  "lighting",
  "tags",
];

/// The scene-analysis task. Construct via [`ImageAnalysisTask::new`].
#[derive(Clone)]
pub struct ImageAnalysisTask {
  schema: Value,
  accept_empty: bool,
}

impl ImageAnalysisTask {
  /// Construct with `accept_empty = false` (a payload that lacks the
  /// required indexable content — `description` AND `tags` both
  /// populated, OR at least one of the substantive detection buckets
  /// `subjects` / `objects` / `actions` non-empty — is treated as a
  /// model regression and rejected; see [`Self::with_accept_empty`]
  /// for the full predicate and the opt-in alternative).
  pub fn new() -> Self {
    Self {
      schema: build_schema(),
      accept_empty: false,
    }
  }

  /// Returns whether the parser accepts payloads that lack the
  /// required indexable content (`description` AND `tags` both
  /// non-empty). See [`Self::with_accept_empty`] for the trade-off.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn accept_empty(&self) -> bool {
    self.accept_empty
  }

  /// Builder-style setter for `accept_empty`.
  ///
  /// When `false` (default), the parser rejects payloads that lack
  /// the required indexable content as [`JsonParseError::NoUsableFields`].
  /// The composite threshold accepts a
  /// payload when **either**:
  ///
  /// - `description` AND `tags` are both populated (the prose +
  ///   keyword path; matches the integration-test smoke criterion),
  ///   OR
  /// - at least one of the **substantive** detection buckets —
  ///   `subjects`, `objects`, or `actions` — is non-empty (the
  ///   substantive-detection path; preserves who/what/where search
  ///   metadata even when the model fails to summarize).
  ///
  /// Style/attribute buckets (`mood`, `lighting`) and single-label
  /// fields (`scene`, `shot_type`) are intentionally NOT in the
  /// substantive path. A payload like `lighting: ["natural light"]`
  /// or `mood: ["calm"]` alone (description and tags empty, no
  /// substantive detections) is more often a regression than a
  /// legitimate weak-but-real scene; rejecting it surfaces the
  /// failure instead of writing a single-attribute stub to the
  /// search index.
  ///
  /// Tags-only, scene-only, description-only, shot_type-only,
  /// mood/lighting-only, and fully-empty payloads all fail both
  /// paths and are rejected. This is the right setting for
  /// indexing pipelines: it surfaces decoder/model regressions that
  /// would otherwise silently overwrite real metadata with sparse
  /// search records.
  ///
  /// When `true`, the parser bypasses the indexable-content check and
  /// returns whatever round-trips through the schema. IMAGE_ANALYSIS_PROMPT
  /// explicitly tells the model to "Use empty arrays or empty strings
  /// when a field is unknown", so on truly low-information frames
  /// (blank, fade-to-black, plain color) compliant model output can
  /// legitimately be sparse or fully-empty. Use this knob if your
  /// pipeline distinguishes "low-information scene" from "no useful
  /// content" via something other than the parser (e.g. scenesdetect's
  /// keyframe scoring).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_accept_empty(mut self, val: bool) -> Self {
    self.accept_empty = val;
    self
  }

  /// In-place setter for `accept_empty`. See
  /// [`Self::with_accept_empty`] for the trade-off.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_accept_empty(&mut self, val: bool) -> &mut Self {
    self.accept_empty = val;
    self
  }
}

impl Default for ImageAnalysisTask {
  fn default() -> Self {
    Self::new()
  }
}

impl Task for ImageAnalysisTask {
  type Output = ImageAnalysis;
  type Value = serde_json::Value;
  type ParseError = llmtask::JsonParseError;

  fn prompt(&self) -> &str {
    IMAGE_ANALYSIS_PROMPT
  }

  fn schema(&self) -> &serde_json::Value {
    &self.schema
  }

  fn grammar(&self) -> llmtask::Grammar {
    // Clone the cached JSON Schema once per call. Cheap relative
    // to constraint compilation, and matches the prior `schema()`
    // contract (which lfm's build_constraint already cloned via
    // TopLevelGrammar::from_json_schema).
    llmtask::Grammar::JsonSchema(self.schema.clone())
  }

  fn parse(&self, raw: &str) -> Result<Self::Output, JsonParseError> {
    let value: Value = serde_json::from_str(raw.trim())?;
    let object = value
      .as_object()
      .ok_or_else(|| JsonParseError::Json(serde::de::Error::custom("expected top-level object")))?;
    let missing = missing_required_fields(object);
    if !missing.is_empty() {
      return Err(JsonParseError::MissingFields(missing));
    }
    let payload: LfmScenePayload = serde_json::from_value(value)?;
    // Indexable-content gate. IMAGE_ANALYSIS_PROMPT instructs the model to "Use
    // empty arrays or empty strings when a field is unknown", so a
    // truly compliant response on a blank/fade-to-black frame can be
    // partially or fully empty. But a decoder/model regression on a
    // normal frame also produces sparse output, and silently
    // overwriting real search metadata with that is worse than
    // failing.
    //
    // Composite threshold:
    // a payload is usable iff EITHER
    //   (a) `description` AND `tags` are both populated (typical
    //       "good" model output, matches the integration-test smoke
    //       criterion), OR
    //   (b) at least one of the substantive detection buckets
    //       (`subjects` / `objects` / `actions`) is non-empty
    //       (the model produced who/what/where evidence even
    //       when prose+keywords are missing).
    //
    // Style/attribute buckets (`mood` / `lighting`) and single-label
    // fields (`scene` / `shot_type`) are intentionally NOT in the
    // substantive path — payloads that populate ONLY those (with
    // description and tags empty) are more often regression signals
    // than legitimate scenes, and writing a single-attribute stub to
    // a search index masks the failure.
    // Callers that distinguish "low-information scene" from
    // "regression" elsewhere opt in via
    // `ImageAnalysisTask::with_accept_empty(true)`.
    if !self.accept_empty && payload.lacks_indexable_content() {
      return Err(JsonParseError::NoUsableFields);
    }
    Ok(payload.into_scene_analysis())
  }
}

fn build_schema() -> Value {
  json!({
      "type": "object",
      "properties": {
          "scene": { "type": "string" },
          "description": { "type": "string" },
          "subjects": { "type": "array", "items": { "type": "string" } },
          "objects": { "type": "array", "items": { "type": "string" } },
          "actions": { "type": "array", "items": { "type": "string" } },
          "mood": { "type": "array", "items": { "type": "string" } },
          "shot_type": { "type": "string" },
          "lighting": { "type": "array", "items": { "type": "string" } },
          "tags": { "type": "array", "items": { "type": "string" } }
      },
      "required": REQUIRED_FIELDS,
      "additionalProperties": false
  })
}

/// Returns required field names that are either absent from the object or
/// present as JSON `null`.
///
/// Both cases violate the schema (every required field must be a string or
/// an array of strings, never null). Treating them identically here matters
/// because the per-field deserializers further down the pipeline silently
/// coerce `null` into the field's default (`Option::None` for strings,
/// empty `DetectionLabels` / `TagList` for arrays). Without this check, a
/// model response like `{"subjects": null, "tags": ["x"], ...}` would be
/// accepted as a valid `ImageAnalysis` with an empty `subjects` list —
/// silently dropping schema-required content and hiding constrained-decoder
/// drift.
fn missing_required_fields(object: &serde_json::Map<String, Value>) -> Vec<&'static str> {
  REQUIRED_FIELDS
    .iter()
    .copied()
    .filter(|field| match object.get(*field) {
      None => true,
      Some(value) => value.is_null(),
    })
    .collect()
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct LfmScenePayload {
  #[serde(default, deserialize_with = "deserialize_optional_trimmed_string")]
  scene: Option<String>,
  #[serde(default, deserialize_with = "deserialize_optional_trimmed_string")]
  description: Option<String>,
  #[serde(default)]
  subjects: DetectionLabels,
  #[serde(default)]
  objects: DetectionLabels,
  #[serde(default)]
  actions: DetectionLabels,
  #[serde(default)]
  mood: DetectionLabels,
  #[serde(default, deserialize_with = "deserialize_optional_single_label")]
  shot_type: Option<String>,
  #[serde(default)]
  lighting: DetectionLabels,
  #[serde(default)]
  tags: TagList,
}

impl LfmScenePayload {
  /// `true` if the payload lacks the minimum content required to
  /// produce a useful indexing record. Composite threshold:
  ///
  /// - **prose+keyword path**: `description` AND `tags` both
  ///   populated — the typical "good" model output that the
  ///   integration test smoke-pins.
  /// - **substantive-detection path**: at least one of the
  ///   substantive detection buckets `subjects` / `objects` /
  ///   `actions` is non-empty — these answer "who/what is in the
  ///   scene and what's happening", which is search-relevant content
  ///   on its own even when prose+keywords are missing.
  ///
  /// Returns `true` (lacks content) when **neither** path holds.
  /// Used by [`ImageAnalysisTask::parse`] to surface model regressions as
  /// [`JsonParseError::NoUsableFields`] unless the caller opts into
  /// `accept_empty = true`.
  ///
  /// **Buckets intentionally excluded from the substantive path:**
  ///
  /// - `mood` / `lighting` — these are style/attribute buckets
  ///   (search filter axes), not standalone content. A regression
  ///   that returns only `lighting: ["natural light"]` or
  ///   `mood: ["calm"]` (description and tags empty, no
  ///   subjects/objects/actions) is more likely a model failure
  ///   than a legitimate "we managed to detect mood but nothing
  ///   else" case, and silently overwriting a richer search record
  ///   with a single-attribute stub is what this gate prevents.
  /// - `scene` / `shot_type` — single-label fields. "Scene-only" or
  ///   "shot_type-only" payloads remain regression signals this
  ///   gate is designed to catch.
  fn lacks_indexable_content(&self) -> bool {
    // `description` deserializes via `deserialize_optional_trimmed_string`
    // which collapses empty/whitespace strings to `None`, so checking
    // `is_none()` suffices.
    let has_prose_and_keywords = self.description.is_some() && !self.tags.0.is_empty();
    let has_substantive_detection =
      !self.subjects.0.is_empty() || !self.objects.0.is_empty() || !self.actions.0.is_empty();
    !has_prose_and_keywords && !has_substantive_detection
  }

  fn into_scene_analysis(self) -> ImageAnalysis {
    // Internal `Option<String>` collapses to `SmolStr` (empty for None).
    // Public `ImageAnalysis` uses empty-string-as-absence to keep the
    // accessor surface simple — see IMAGE_ANALYSIS_PROMPT, which already
    // instructs the model to emit empty strings for unknown fields.
    let to_labels =
      |list: DetectionLabels| -> Vec<SmolStr> { list.0.into_iter().map(SmolStr::from).collect() };
    ImageAnalysis::new()
      .with_scene(self.scene.map(SmolStr::from).unwrap_or_default())
      .with_description(self.description.map(SmolStr::from).unwrap_or_default())
      .with_subjects(to_labels(self.subjects))
      .with_objects(to_labels(self.objects))
      .with_actions(to_labels(self.actions))
      .with_mood(to_labels(self.mood))
      .with_shot_type(self.shot_type.map(SmolStr::from).unwrap_or_default())
      .with_lighting(to_labels(self.lighting))
      .with_tags(self.tags.0.into_iter().map(SmolStr::from).collect())
  }
}

/// Used for the `tags` field. The string fallback is split on commas /
/// semicolons / newlines, because tag-list drift (model dropped the
/// array around a flat comma-separated string) is the historically
/// common case for that field.
#[derive(Debug, Default)]
struct TagList(Vec<String>);

impl<'de> Deserialize<'de> for TagList {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
      String(String),
      List(Vec<String>),
    }

    let raw = Option::<Repr>::deserialize(deserializer)?;
    let mut values = Vec::new();
    match raw {
      // String fallback: model returned a flattened comma/semicolon/newline-
      // separated tag string instead of an array (real production drift
      // for the tags field).
      Some(Repr::String(value)) => push_string_list_items(&mut values, &value),
      // Array form: trim and dedupe each element verbatim. Do NOT split
      // on commas — a tag like `"july 4, 2026"` must stay one entry.
      Some(Repr::List(items)) => {
        for item in items {
          push_array_item(&mut values, item);
        }
      }
      None => {}
    }
    Ok(Self(values))
  }
}

/// Used for detection-array fields (`subjects`, `objects`, `actions`,
/// `mood`, `lighting`). Detection labels can naturally contain commas
/// (e.g. `"red, white, and blue flag"`, `"middle-aged man in red
/// jacket, sunglasses"`). String-fallback splitting was wrong for
/// these fields — model drift could otherwise turn one
/// comma-bearing label into three bogus detections.
///
/// Behavior:
/// - JSON array: trim and dedupe each element verbatim (no splitting).
/// - JSON string: treat as a single-element list. Single label, no
///   comma-split. This preserves the data when the constrained
///   decoder drifts to a scalar string (rare with `JsonSchema`
///   constraint but defensive).
/// - JSON null / missing: empty list.
#[derive(Debug, Default)]
struct DetectionLabels(Vec<String>);

impl<'de> Deserialize<'de> for DetectionLabels {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
      String(String),
      List(Vec<String>),
    }

    let raw = Option::<Repr>::deserialize(deserializer)?;
    let mut values = Vec::new();
    match raw {
      // Single string → one detection label, no splitting.
      Some(Repr::String(value)) => push_array_item(&mut values, value),
      Some(Repr::List(items)) => {
        for item in items {
          push_array_item(&mut values, item);
        }
      }
      None => {}
    }
    Ok(Self(values))
  }
}

fn push_array_item(values: &mut Vec<String>, raw: String) {
  let trimmed = raw.trim();
  if !trimmed.is_empty() && !values.iter().any(|existing| existing == trimmed) {
    values.push(trimmed.to_owned());
  }
}

fn push_string_list_items(values: &mut Vec<String>, raw: &str) {
  for part in raw.split([',', ';', '\n']) {
    let part = part.trim();
    if !part.is_empty() && !values.iter().any(|existing| existing == part) {
      values.push(part.to_owned());
    }
  }
}

fn deserialize_optional_trimmed_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  Ok(Option::<String>::deserialize(deserializer)?.and_then(normalize_string))
}

fn deserialize_optional_single_label<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  #[derive(Deserialize)]
  #[serde(untagged)]
  enum Repr {
    String(String),
    List(Vec<String>),
  }

  match Option::<Repr>::deserialize(deserializer)? {
    Some(Repr::String(value)) => Ok(normalize_string(value)),
    Some(Repr::List(values)) => {
      let mut normalized = values.into_iter().filter_map(normalize_string);
      let first = normalized.next();
      if normalized.next().is_some() {
        return Err(serde::de::Error::custom(
          "expected a single shot_type label, got multiple values",
        ));
      }
      Ok(first)
    }
    None => Ok(None),
  }
}

fn normalize_string(value: String) -> Option<String> {
  let trimmed = value.trim();
  (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

#[cfg(test)]
mod tests {
  use smol_str::SmolStr;

  use super::*;

  /// `IMAGE_ANALYSIS_PROMPT` must not enumerate value tokens.
  /// mistralrs 0.8 applies `presence_penalty` over
  /// `seq.get_toks()` (prompt + generated), so any value-token in
  /// the prompt gets a `-presence_penalty`
  /// logit shift before generation — biasing the model away from
  /// legitimate matches in the deterministic-mode default. Format
  /// guidance must use descriptive constraints (word counts,
  /// lowercase, etc.) instead of enumerated examples.
  ///
  /// This guard catches accidental regressions (someone copy-pastes
  /// a new field-instruction line that includes `e.g. "..."`
  /// examples, or reverts to an older prompt). It is not a defense
  /// against deliberate edits — a determined reverter can also
  /// remove tokens from this list.
  #[test]
  fn scene_prompt_does_not_enumerate_value_tokens() {
    let prompt_lower = IMAGE_ANALYSIS_PROMPT.to_lowercase();
    // Distinctive multi-word phrases (and a few unambiguous single
    // words) that appeared in older prompts' `e.g.` enumerations
    // across the nine field-instruction lines.
    let banned_tokens = [
      "stage performance",
      "middle-aged man",
      "golden retriever",
      "birthday cake",
      "vintage red sports car",
      "cutting cake",
      "taking photos",
      "wide shot",
      "close-up",
      "medium shot",
      "over-the-shoulder",
      "celebratory",
      "natural light",
      "low light",
      "backlit",
    ];
    for token in banned_tokens {
      assert!(
        !prompt_lower.contains(&token.to_lowercase()),
        "IMAGE_ANALYSIS_PROMPT must not enumerate value token {token:?} \
                 (prompt-vocabulary tokens get \
                 -presence_penalty logit shift in deterministic mode); \
                 use descriptive format guidance (word counts, lowercase) \
                 instead of `e.g. \"...\"` examples"
      );
    }
  }

  #[test]
  fn parse_valid_json() {
    let json = r#"{"scene":"beach","description":"Sunset over the ocean","subjects":["person"],"objects":["sun"],"actions":["watching"],"mood":["calm"],"shot_type":"wide shot","lighting":["golden hour"],"tags":["sunset","ocean"]}"#;
    let task = ImageAnalysisTask::new();
    let result = task.parse(json).expect("parse should succeed");
    assert_eq!(result.scene(), "beach");
    assert_eq!(result.description(), "Sunset over the ocean");
    assert_eq!(result.mood().len(), 1);
    assert_eq!(result.subjects().len(), 1);
  }

  #[test]
  fn reject_json_with_wrapper_text() {
    let text =
      "Here is the analysis:\n{\"scene\":\"office\",\"description\":\"People working\"}\nDone.";
    let task = ImageAnalysisTask::new();
    assert!(task.parse(text).is_err());
  }

  #[test]
  fn reject_plain_text_output() {
    let text = "A beautiful sunset over the ocean.";
    let task = ImageAnalysisTask::new();
    assert!(task.parse(text).is_err());
  }

  #[test]
  fn parse_comma_separated_tag_string() {
    let json = r#"{"scene":"stage performance","description":"A singer on stage","subjects":[],"objects":["microphone"],"actions":["singing"],"mood":["energetic"],"shot_type":"medium shot","lighting":["spotlight"],"tags":"concert, live music, spotlight"}"#;
    let task = ImageAnalysisTask::new();
    let result = task.parse(json).expect("parse should succeed");
    assert_eq!(
      result.tags(),
      &[
        SmolStr::from("concert"),
        SmolStr::from("live music"),
        SmolStr::from("spotlight"),
      ][..]
    );
  }

  #[test]
  fn reject_empty_json_payload() {
    let task = ImageAnalysisTask::new();
    assert!(task.parse("{}").is_err());
  }

  #[test]
  fn reject_unknown_json_fields() {
    let json = r#"{"description":"A singer on stage","extra":"unexpected"}"#;
    let task = ImageAnalysisTask::new();
    assert!(task.parse(json).is_err());
  }

  #[test]
  fn reject_missing_required_fields() {
    let json = r#"{"description":"A singer on stage","tags":["concert"]}"#;
    let task = ImageAnalysisTask::new();
    assert!(task.parse(json).is_err());
  }

  #[test]
  fn parse_array_form_subjects() {
    // Array form: each element becomes one detection, no splitting.
    let json_list = r#"{"scene":"x","description":"y","subjects":["a","b"],"objects":[],"actions":[],"mood":[],"shot_type":"x","lighting":[],"tags":["t"]}"#;
    let task = ImageAnalysisTask::new();
    let result = task.parse(json_list).expect("list-form parse");
    assert_eq!(result.subjects().len(), 2);
    assert_eq!(result.subjects()[0], "a");
    assert_eq!(result.subjects()[1], "b");
  }

  #[test]
  fn subjects_string_form_treated_as_single_label() {
    // Previously the string-fallback branch of StringList split
    // scalar strings on commas, so a model drift to `"subjects":
    // "red, white, and blue flag"` was silently turned into three
    // bogus detections ("red", "white", "and blue
    // flag"). The fix uses a separate `DetectionLabels` deserializer for
    // detection-array fields that wraps the string as a single label —
    // detection labels can naturally contain commas. (`tags` keeps
    // comma-split behavior; that field is the historically common
    // tag-list-as-string drift case and tests it separately.)
    let json = r#"{"scene":"x","description":"y","subjects":"middle-aged man, in red jacket","objects":[],"actions":[],"mood":[],"shot_type":"x","lighting":[],"tags":["t"]}"#;
    let task = ImageAnalysisTask::new();
    let result = task.parse(json).expect("string-form parse");
    assert_eq!(
      result.subjects().len(),
      1,
      "string-form must wrap as a single label, not comma-split"
    );
    assert_eq!(result.subjects()[0], "middle-aged man, in red jacket");
  }

  #[test]
  fn reject_all_required_fields_empty_payload_by_default() {
    let json = r#"{
          "scene": "",
          "description": "",
          "subjects": [],
          "objects": [],
          "actions": [],
          "mood": [],
          "shot_type": "",
          "lighting": [],
          "tags": []
        }"#;
    let task = ImageAnalysisTask::new();
    let err = task
      .parse(json)
      .expect_err("default ImageAnalysisTask must reject all-empty payload");
    assert!(
      matches!(err, JsonParseError::NoUsableFields),
      "expected NoUsableFields, got {err:?}"
    );
  }

  #[test]
  fn accept_all_required_fields_empty_payload_when_opted_in() {
    let json = r#"{
          "scene": "",
          "description": "",
          "subjects": [],
          "objects": [],
          "actions": [],
          "mood": [],
          "shot_type": "",
          "lighting": [],
          "tags": []
        }"#;
    let task = ImageAnalysisTask::new().with_accept_empty(true);
    let result = task
      .parse(json)
      .expect("opt-in must accept the all-empty payload");
    assert!(result.scene().is_empty());
    assert!(result.description().is_empty());
    assert!(result.subjects().is_empty());
    assert!(result.objects().is_empty());
    assert!(result.actions().is_empty());
    assert!(result.mood().is_empty());
    assert!(result.shot_type().is_empty());
    assert!(result.lighting().is_empty());
    assert!(result.tags().is_empty());
  }

  #[test]
  fn reject_tags_only_payload_by_default() {
    let json = r#"{
          "scene": "",
          "description": "",
          "subjects": [],
          "objects": [],
          "actions": [],
          "mood": [],
          "shot_type": "",
          "lighting": [],
          "tags": ["concert", "live music"]
        }"#;
    let task = ImageAnalysisTask::new();
    let err = task
      .parse(json)
      .expect_err("default ImageAnalysisTask must reject tags-only payload");
    assert!(
      matches!(err, JsonParseError::NoUsableFields),
      "expected NoUsableFields, got {err:?}"
    );
  }

  #[test]
  fn reject_scene_only_payload_by_default() {
    let json = r#"{
          "scene": "office",
          "description": "",
          "subjects": [],
          "objects": [],
          "actions": [],
          "mood": [],
          "shot_type": "",
          "lighting": [],
          "tags": []
        }"#;
    let task = ImageAnalysisTask::new();
    let err = task
      .parse(json)
      .expect_err("default ImageAnalysisTask must reject scene-only payload");
    assert!(
      matches!(err, JsonParseError::NoUsableFields),
      "expected NoUsableFields, got {err:?}"
    );
  }

  #[test]
  fn reject_description_only_payload_by_default() {
    let json = r#"{
          "scene": "",
          "description": "People working in an office",
          "subjects": [],
          "objects": [],
          "actions": [],
          "mood": [],
          "shot_type": "",
          "lighting": [],
          "tags": []
        }"#;
    let task = ImageAnalysisTask::new();
    let err = task
      .parse(json)
      .expect_err("default ImageAnalysisTask must reject description-only payload");
    assert!(
      matches!(err, JsonParseError::NoUsableFields),
      "expected NoUsableFields, got {err:?}"
    );
  }

  #[test]
  fn accept_minimal_indexable_payload() {
    let json = r#"{
          "scene": "",
          "description": "Two people talking",
          "subjects": [],
          "objects": [],
          "actions": [],
          "mood": [],
          "shot_type": "",
          "lighting": [],
          "tags": ["conversation"]
        }"#;
    let task = ImageAnalysisTask::new();
    let result = task
      .parse(json)
      .expect("description+tags must clear the indexable threshold");
    assert_eq!(result.description(), "Two people talking");
    assert_eq!(result.tags(), &[SmolStr::from("conversation")][..]);
    assert!(result.subjects().is_empty());
    assert!(result.objects().is_empty());
    assert!(result.scene().is_empty());
  }

  #[test]
  fn accept_detection_rich_payload_with_empty_description_and_tags() {
    let json = r#"{
          "scene": "",
          "description": "",
          "subjects": ["middle-aged woman in red dress"],
          "objects": ["wedding cake"],
          "actions": ["cutting cake"],
          "mood": [],
          "shot_type": "",
          "lighting": [],
          "tags": []
        }"#;
    let task = ImageAnalysisTask::new();
    let result = task.parse(json).expect(
      "detection-rich payload must clear the indexable threshold via \
             the detection-bucket path even when description+tags are empty",
    );
    assert_eq!(result.subjects().len(), 1);
    assert_eq!(result.objects().len(), 1);
    assert_eq!(result.actions().len(), 1);
    assert!(result.description().is_empty());
    assert!(result.tags().is_empty());
  }

  #[test]
  fn accept_subjects_only_payload() {
    let json = r#"{
          "scene": "",
          "description": "",
          "subjects": ["a single subject label"],
          "objects": [],
          "actions": [],
          "mood": [],
          "shot_type": "",
          "lighting": [],
          "tags": []
        }"#;
    let task = ImageAnalysisTask::new();
    let result = task
      .parse(json)
      .expect("subjects-only must clear the indexable threshold");
    assert_eq!(result.subjects().len(), 1);
  }

  #[test]
  fn accept_objects_only_payload() {
    let json = r#"{
          "scene": "",
          "description": "",
          "subjects": [],
          "objects": ["a single object label"],
          "actions": [],
          "mood": [],
          "shot_type": "",
          "lighting": [],
          "tags": []
        }"#;
    let task = ImageAnalysisTask::new();
    let result = task
      .parse(json)
      .expect("objects-only must clear the indexable threshold");
    assert_eq!(result.objects().len(), 1);
  }

  #[test]
  fn accept_actions_only_payload() {
    let json = r#"{
          "scene": "",
          "description": "",
          "subjects": [],
          "objects": [],
          "actions": ["a single action label"],
          "mood": [],
          "shot_type": "",
          "lighting": [],
          "tags": []
        }"#;
    let task = ImageAnalysisTask::new();
    let result = task
      .parse(json)
      .expect("actions-only must clear the indexable threshold");
    assert_eq!(result.actions().len(), 1);
  }

  #[test]
  fn reject_mood_only_payload_by_default() {
    let json = r#"{
          "scene": "",
          "description": "",
          "subjects": [],
          "objects": [],
          "actions": [],
          "mood": ["calm"],
          "shot_type": "",
          "lighting": [],
          "tags": []
        }"#;
    let task = ImageAnalysisTask::new();
    let err = task
      .parse(json)
      .expect_err("default ImageAnalysisTask must reject mood-only payload");
    assert!(
      matches!(err, JsonParseError::NoUsableFields),
      "expected NoUsableFields, got {err:?}"
    );
  }

  #[test]
  fn reject_lighting_only_payload_by_default() {
    let json = r#"{
          "scene": "",
          "description": "",
          "subjects": [],
          "objects": [],
          "actions": [],
          "mood": [],
          "shot_type": "",
          "lighting": ["natural light"],
          "tags": []
        }"#;
    let task = ImageAnalysisTask::new();
    let err = task
      .parse(json)
      .expect_err("default ImageAnalysisTask must reject lighting-only payload");
    assert!(
      matches!(err, JsonParseError::NoUsableFields),
      "expected NoUsableFields, got {err:?}"
    );
  }

  #[test]
  fn reject_attribute_only_payload_by_default() {
    let json = r#"{
          "scene": "",
          "description": "",
          "subjects": [],
          "objects": [],
          "actions": [],
          "mood": ["tense"],
          "shot_type": "",
          "lighting": ["low light"],
          "tags": []
        }"#;
    let task = ImageAnalysisTask::new();
    let err = task
      .parse(json)
      .expect_err("style-attribute-only payload must reject regardless of bucket count");
    assert!(
      matches!(err, JsonParseError::NoUsableFields),
      "expected NoUsableFields, got {err:?}"
    );
  }

  #[test]
  fn reject_null_required_array() {
    let json = r#"{
          "scene": "office",
          "description": "people working",
          "subjects": null,
          "objects": [],
          "actions": [],
          "mood": [],
          "shot_type": "wide",
          "lighting": [],
          "tags": ["work"]
        }"#;
    let task = ImageAnalysisTask::new();
    let err = task
      .parse(json)
      .expect_err("null required field must be rejected");
    match err {
      JsonParseError::MissingFields(fields) => {
        assert!(
          fields.contains(&"subjects"),
          "expected 'subjects' in MissingFields, got {fields:?}"
        );
      }
      other => panic!("expected MissingFields, got {other:?}"),
    }
  }

  #[test]
  fn reject_null_required_string() {
    let json = r#"{
          "scene": null,
          "description": "people working",
          "subjects": ["person"],
          "objects": [],
          "actions": [],
          "mood": [],
          "shot_type": "wide",
          "lighting": [],
          "tags": ["work"]
        }"#;
    let task = ImageAnalysisTask::new();
    let err = task
      .parse(json)
      .expect_err("null required field must be rejected");
    match err {
      JsonParseError::MissingFields(fields) => {
        assert!(
          fields.contains(&"scene"),
          "expected 'scene' in MissingFields, got {fields:?}"
        );
      }
      other => panic!("expected MissingFields, got {other:?}"),
    }
  }

  #[test]
  fn reject_multiple_null_required_fields() {
    let json = r#"{
          "scene": null,
          "description": null,
          "subjects": null,
          "objects": [],
          "actions": [],
          "mood": [],
          "shot_type": "wide",
          "lighting": [],
          "tags": ["work"]
        }"#;
    let task = ImageAnalysisTask::new();
    let err = task
      .parse(json)
      .expect_err("null required fields must be rejected");
    match err {
      JsonParseError::MissingFields(fields) => {
        assert!(fields.contains(&"scene"), "missing 'scene' in {fields:?}");
        assert!(
          fields.contains(&"description"),
          "missing 'description' in {fields:?}"
        );
        assert!(
          fields.contains(&"subjects"),
          "missing 'subjects' in {fields:?}"
        );
      }
      other => panic!("expected MissingFields, got {other:?}"),
    }
  }

  #[test]
  fn array_elements_are_not_comma_split() {
    let json = r#"{
          "scene": "patriotic event",
          "description": "Flag display",
          "subjects": ["middle-aged man, in red jacket"],
          "objects": ["red, white, and blue flag", "birthday cake with candles, balloons"],
          "actions": ["waving"],
          "mood": ["festive"],
          "shot_type": "wide shot",
          "lighting": ["natural, dramatic backlight"],
          "tags": ["july 4, 2026"]
        }"#;
    let task = ImageAnalysisTask::new();
    let result = task.parse(json).expect("parse should succeed");
    assert_eq!(result.subjects().len(), 1);
    assert_eq!(result.subjects()[0], "middle-aged man, in red jacket");
    assert_eq!(result.objects().len(), 2);
    assert_eq!(result.objects()[0], "red, white, and blue flag");
    assert_eq!(result.objects()[1], "birthday cake with candles, balloons");
    assert_eq!(result.lighting().len(), 1);
    assert_eq!(result.lighting()[0], "natural, dramatic backlight");
    assert_eq!(result.tags().len(), 1);
    assert_eq!(result.tags()[0].as_str(), "july 4, 2026");
  }

  #[test]
  fn parse_shot_type_list_form() {
    // shot_type accepts the list form `["wide shot"]` (one element)
    // via `deserialize_optional_single_label`.
    let json_one = r#"{"scene":"x","description":"y","subjects":[],"objects":[],"actions":[],"mood":[],"shot_type":["wide shot"],"lighting":[],"tags":["t"]}"#;
    let task = ImageAnalysisTask::new();
    let result = task.parse(json_one).expect("single-element list parse");
    assert_eq!(result.shot_type(), "wide shot");

    // Multi-element list is rejected.
    let json_many = r#"{"scene":"x","description":"y","subjects":[],"objects":[],"actions":[],"mood":[],"shot_type":["wide","close-up"],"lighting":[],"tags":["t"]}"#;
    assert!(task.parse(json_many).is_err());
  }
}
