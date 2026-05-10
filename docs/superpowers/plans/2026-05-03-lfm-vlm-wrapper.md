# LFM2.5-VL ONNX Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `lfm` Rust crate that wraps LFM2.5-VL-450M-ONNX with a sync `Engine` + `Task` trait + free-form `generate`, sharing the `SceneAnalysis` data type with `qwen` via a new `vlm-tasks` crate.

**Architecture:** 3-graph ONNX inference (vision_encoder, embed_tokens, decoder_model_merged) via raw `ort 2.0.0-rc.12`. **Per-image vision encoder calls** to work around the Phase 0 Gate B silent-corruption finding (multi-image batched calls produce wrong embeddings). `llguidance` for schema-constrained sampling on the structured-output path. `minijinja` for chat-template rendering with a custom no-op `generation` tag.

**Tech Stack:** Rust 2024 edition (rust-version 1.95), ort 2.0.0-rc.12, tokenizers 0.23, llguidance 1.7, minijinja 2, image 0.25, smol_str 0.3, thiserror 2, tracing 0.1.

**Spec:** `lfm/docs/superpowers/specs/2026-05-03-lfm-vlm-wrapper-design.md`

**Workspace structure (pinned — option A: independent repos with path deps):** The three crates (`vlm-tasks/`, `qwen/`, `lfm/`) are independent git repositories under `findit-studio/`. They use Cargo path deps (`vlm-tasks = { path = "../vlm-tasks" }`) — NOT a Cargo workspace, NOT git submodules. **None are published to crates.io** (qwen + lfm READMEs explicitly say "Internal findit-studio crate. Not published"). This matches the existing siglip2/egemma/qwen layout, avoids workspace-wide rebuilds, and eliminates the publishability dependency chain (vlm-tasks would otherwise have to publish first). If that ever changes, the migration to a workspace is a one-line change at `findit-studio/Cargo.toml`; for v0 we stay independent.

**Phase 0 fixtures (already in repo, verified):**
- `lfm/tests/fixtures/onnx_io_contract.json` — G1–G5 resolved (decoder has no `position_ids`; conv-cache uses sparse layer indices `[0,1,3,4,6,7,9,11,13,15]`; vision output is `image_features`; vision input is pre-patchified `[batch, num_patches, 768]`)
- `lfm/tests/fixtures/multi_image_ordering_proof.json` — G6 RESOLVED (FAILED → per-image vision calls required)

---

## File Structure

Workspace layout (siblings under `findit-studio/`):

```
vlm-tasks/                       # NEW CRATE (Phase 1)
├── Cargo.toml
├── README.md
├── LICENSE-MIT, LICENSE-APACHE
└── src/
    ├── lib.rs                   # pub use {task::*, scene::*}
    ├── task.rs                  # Task trait + ParseError
    └── scene.rs                 # SceneAnalysis (data type only)

qwen/                            # EXISTING — minor migration (Phase 2)
├── Cargo.toml                   # add vlm-tasks dep
└── src/
    ├── lib.rs                   # add re-exports
    ├── task.rs                  # remove (moved to vlm-tasks)
    └── scene.rs                 # SceneAnalysis becomes re-export

lfm/                             # NEW IMPLEMENTATION (Phase 3)
├── Cargo.toml                   # rewrite from template-rs
├── README.md, CHANGELOG.md
├── build.rs                     # template default; no-op
├── models/                      # whitelisted by Cargo.toml include
│   ├── tokenizer.json           # bundle-feature-gated, 4.5 MB
│   ├── chat_template.jinja      # always shipped, 3.8 KB
│   └── preprocessor_config.json # build-fixture only, 0.7 KB
├── src/
│   ├── lib.rs                   # re-exports + features + BUNDLED_* consts
│   ├── error.rs                 # Error enum + named constructors
│   ├── options.rs               # RequestOptions, ImageBudget, ThreadOptions, Options
│   ├── embedding.rs             # Embedding (post-projector vision-tile vector)
│   ├── chat_template.rs         # apply_chat_template + expand_image_placeholders + tokens
│   ├── preproc/
│   │   ├── mod.rs               # Preprocessor + PreprocessedImage + EXIF helpers
│   │   └── tile_grid.rs         # find_closest_aspect_ratio + smart_resize
│   ├── runtime/                 # gated on `inference` feature
│   │   ├── mod.rs               # re-exports
│   │   ├── session.rs           # build_session + check_outlet + validate_*_session
│   │   ├── vision.rs            # VisionEncoder (single-image)
│   │   ├── embed_tokens.rs      # EmbedTokens
│   │   ├── decoder.rs           # Decoder + KvCache (sparse indices)
│   │   └── sampler.rs           # Sampler trait + FreeSampler + ConstrainedSampler
│   ├── generate.rs              # end-to-end pipeline (per-image vision calls)
│   ├── engine.rs                # public Engine
│   ├── task.rs                  # re-exports of vlm_tasks::*
│   └── scene.rs                 # lfm-specific SceneTask impl
├── examples/
│   ├── smoke.rs                 # phase-zero "does it work"
│   ├── scene_analysis.rs
│   ├── preprocess_only.rs       # wasm-compat showcase
│   └── qwen_compare.rs
├── benches/
│   ├── bench_preproc.rs
│   ├── bench_tile_grid.rs
│   └── bench_chat_template.rs
├── scripts/                     # ALREADY IN REPO from Phase 0
│   ├── capture_onnx_io.py
│   ├── verify_multi_image_ordering.py
│   └── README.md
└── tests/
    ├── fixtures/                # Phase-0 JSONs already present; add airport images + parity fixtures
    │   ├── onnx_io_contract.json                    # ALREADY HERE
    │   ├── multi_image_ordering_proof.json          # ALREADY HERE
    │   ├── airport_01.jpg, airport_02.jpg, airport_03.jpg  # port from qwen
    │   ├── chat_template_cases.json
    │   ├── tile_grid_cases.json
    │   ├── image_expansion_cases.json
    │   └── scene_payloads/                          # canonical / drift / null cases
    └── integration.rs           # gated on `integration` feature
```

---

## Phase 1: vlm-tasks crate

### Task 1: Create vlm-tasks crate (one task — small surface)

**Files:**
- Create: `vlm-tasks/Cargo.toml`
- Create: `vlm-tasks/src/lib.rs`
- Create: `vlm-tasks/src/task.rs`
- Create: `vlm-tasks/src/scene.rs`
- Create: `vlm-tasks/README.md`
- Create: `vlm-tasks/LICENSE-MIT`, `vlm-tasks/LICENSE-APACHE` (copy from qwen)

- [ ] **Step 1: Scaffold the directory and copy licenses**

```bash
mkdir -p /Users/user/Develop/findit-studio/vlm-tasks/src
cp /Users/user/Develop/findit-studio/qwen/LICENSE-MIT /Users/user/Develop/findit-studio/vlm-tasks/
cp /Users/user/Develop/findit-studio/qwen/LICENSE-APACHE /Users/user/Develop/findit-studio/vlm-tasks/
```

- [ ] **Step 2: Write Cargo.toml**

Create `vlm-tasks/Cargo.toml` with:

```toml
[package]
name         = "vlm-tasks"
version      = "0.1.0"
edition      = "2024"
rust-version = "1.95"
description  = "Shared types for findit-studio VLM engines: Task trait, ParseError, SceneAnalysis"
license      = "MIT OR Apache-2.0"

[dependencies]
serde      = { version = "1", features = ["derive"], optional = true }
serde_json = "1"
smol_str   = "0.3"
thiserror  = "2"

[features]
default = []
serde   = ["dep:serde", "smol_str/serde"]

[lints.rust]
rust_2018_idioms     = "warn"
single_use_lifetimes = "warn"
unexpected_cfgs      = { level = "warn", check-cfg = ['cfg(docsrs)'] }
```

- [ ] **Step 3: Write the failing tests for `Task` and `ParseError`**

Create `vlm-tasks/src/task.rs`:

```rust
//! `Task` trait and `ParseError` — the cross-engine abstraction.

use serde_json::Value;

/// A structured-output task description.
///
/// Implementations supply the prompt, the JSON schema for constrained
/// decoding, and a parser that turns the model's raw text into a typed
/// `Output`. The trait is `Send + Sync` and `Output: Send` so trait
/// objects (`dyn Task<Output = ...>`) and concurrent call sites work
/// without extra bounds at the call site.
pub trait Task: Send + Sync {
  /// The typed result of a successful run.
  type Output: Send;

  /// The user-message prompt sent alongside the images.
  fn prompt(&self) -> &str;

  /// JSON schema used for constrained decoding.
  fn schema(&self) -> &Value;

  /// Parse the model's raw text output into a typed `Output`.
  fn parse(&self, raw: &str) -> Result<Self::Output, ParseError>;
}

/// Errors returned by [`Task::parse`].
#[derive(thiserror::Error, Debug)]
pub enum ParseError {
  /// `serde_json` failed to parse the response as valid JSON.
  #[error("invalid JSON: {0}")]
  Json(#[from] serde_json::Error),
  /// JSON parsed but one or more required schema fields are absent or
  /// present as JSON `null`.
  #[error("schema violation: required fields missing or null: {0:?}")]
  MissingFields(Vec<&'static str>),
  /// JSON parsed and had no missing fields, but every value was empty.
  #[error("structured response had no usable fields")]
  NoUsableFields,
}

#[cfg(test)]
mod tests {
  use super::*;

  /// `Task` is dyn-compatible with `Output` carrying through.
  #[test]
  fn task_is_dyn_compatible() {
    struct Dummy;
    impl Task for Dummy {
      type Output = ();
      fn prompt(&self) -> &str { "" }
      fn schema(&self) -> &Value { static V: once_cell::sync::Lazy<Value> = once_cell::sync::Lazy::new(|| Value::Null); &V }
      fn parse(&self, _raw: &str) -> Result<(), ParseError> { Ok(()) }
    }
    let _: Box<dyn Task<Output = ()>> = Box::new(Dummy);
  }
}
```

(Note: `once_cell` would be a dev-dep just for this test. Alternative — use a free function returning a `&'static Value` via `OnceLock`. Pick whichever is cleaner.)

- [ ] **Step 4: Write the failing tests for `SceneAnalysis`**

Create `vlm-tasks/src/scene.rs` — port verbatim from `qwen/src/scene.rs::SceneAnalysis` (the data type and its accessors only — NOT the SceneTask, prompt, schema, or parser, all of which stay in each engine crate). Specifically:

```rust
//! `SceneAnalysis` — the canonical scene-analysis output type, shared
//! across `qwen` and `lfm` engines. Each engine's `SceneTask` constructs
//! values of this type; downstream consumers can pass `&SceneAnalysis`
//! references between engine outputs without conversion.

use smol_str::SmolStr;

/// Structured scene-level VLM output. Construct via an engine's
/// `SceneTask::parse` (the `Task::parse` impl) or, for tests/builders,
/// `SceneAnalysis::new` followed by `with_*` chains.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SceneAnalysis {
  scene: SmolStr,
  description: SmolStr,
  subjects: Vec<SmolStr>,
  objects: Vec<SmolStr>,
  actions: Vec<SmolStr>,
  mood: Vec<SmolStr>,
  shot_type: SmolStr,
  lighting: Vec<SmolStr>,
  tags: Vec<SmolStr>,
}

impl SceneAnalysis {
  /// Construct an empty `SceneAnalysis` (all fields default).
  pub fn new() -> Self { Self::default() }

  // Accessor surface — port the full {scene/description/subjects/objects/
  // actions/mood/shot_type/lighting/tags} × {getter, with_*, set_*} block
  // from qwen/src/scene.rs lines 67-275 verbatim.
  // Each field gets:
  //   pub fn FIELD(&self) -> &str / &[SmolStr]
  //   pub fn with_FIELD(mut self, val: ...) -> Self
  //   pub fn set_FIELD(&mut self, val: ...) -> &mut Self
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_is_empty() {
    let s = SceneAnalysis::new();
    assert!(s.scene().is_empty());
    assert!(s.description().is_empty());
    assert!(s.subjects().is_empty());
    assert_eq!(s, SceneAnalysis::default());
  }

  #[test]
  fn builder_chains() {
    let s = SceneAnalysis::new()
      .with_scene("airport")
      .with_description("travelers walking through terminal")
      .with_subjects(vec!["middle-aged woman".into(), "child".into()])
      .with_tags(vec!["airport".into(), "travel".into(), "indoor".into()]);
    assert_eq!(s.scene(), "airport");
    assert_eq!(s.subjects().len(), 2);
    assert_eq!(s.tags().len(), 3);
  }

  #[test]
  fn set_in_place() {
    let mut s = SceneAnalysis::new();
    s.set_scene("plaza");
    assert_eq!(s.scene(), "plaza");
  }
}
```

- [ ] **Step 5: Write `lib.rs`**

Create `vlm-tasks/src/lib.rs`:

```rust
//! Shared types for findit-studio VLM engines.
//!
//! This crate hosts the cross-engine abstractions that both `qwen` and
//! `lfm` depend on: the [`Task`] trait, [`ParseError`], and the
//! canonical [`SceneAnalysis`] data type. Each engine ships its own
//! `SceneTask` implementation — the prompt, schema, and parser are
//! engine-specific (tuned to each model's drift patterns) — but they
//! all produce values of the same `SceneAnalysis` type.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes, missing_docs)]

pub mod scene;
pub mod task;

pub use scene::SceneAnalysis;
pub use task::{ParseError, Task};
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /Users/user/Develop/findit-studio/vlm-tasks
cargo test --lib
```

Expected: all tests pass (default/empty, builder chains, set in place, dyn-compatible).

- [ ] **Step 7: Run clippy to verify lints**

```bash
cd /Users/user/Develop/findit-studio/vlm-tasks
cargo clippy --lib --all-targets -- -D warnings
```

Expected: no warnings.

- [ ] **Step 8: Write README + CHANGELOG**

Create `vlm-tasks/README.md`:

```markdown
# vlm-tasks

Shared types for findit-studio VLM engines.

This crate hosts the cross-engine abstractions that both [`qwen`] and
[`lfm`] depend on: the `Task` trait, `ParseError`, and the canonical
`SceneAnalysis` data type. Each engine ships its own `SceneTask`
implementation — prompt, schema, and parser are engine-specific —
but they all produce values of the same `SceneAnalysis` type.

## Status

Internal findit-studio crate. Not published to crates.io.

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE).
```

- [ ] **Step 9: Commit**

```bash
cd /Users/user/Develop/findit-studio
# vlm-tasks doesn't have its own git repo — initialize one
cd vlm-tasks
git init
git add .
git commit -m "$(cat <<'EOF'
Initial vlm-tasks crate: Task trait, ParseError, SceneAnalysis

Shared types for findit-studio VLM engines (qwen, lfm). Hosts the
cross-engine abstractions:
- Task trait (prompt + schema + parse)
- ParseError (Json, MissingFields, NoUsableFields)
- SceneAnalysis (data type only — accessors and builders)

Each engine's SceneTask impl stays in that engine (parsers are tuned
per-model). Downstream consumers get one canonical SceneAnalysis type
that flows between qwen and lfm without conversion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: qwen migration

### Task 2: Migrate qwen to depend on vlm-tasks

**Files:**
- Modify: `qwen/Cargo.toml` (add `vlm-tasks` dep)
- Modify: `qwen/src/lib.rs` (re-exports, drop owned Task)
- Modify: `qwen/src/task.rs` → DELETE (moved to vlm-tasks)
- Modify: `qwen/src/scene.rs` (replace `SceneAnalysis` definition with re-export of `vlm_tasks::SceneAnalysis`; keep `SceneTask` impl)
- Verify: `cargo test --lib` and `cargo test --features integration` (if env var set) still pass

- [ ] **Step 1: Add vlm-tasks dep**

Modify `qwen/Cargo.toml`:

```toml
[dependencies]
mistralrs  = { version = "0.8", features = ["metal"] }
image      = { version = "0.25", default-features = false, features = ["jpeg"] }
serde      = { version = "1", features = ["derive"] }
serde_json = "1"
smol_str   = "0.3"
thiserror  = "2"
tracing    = "0.1"
vlm-tasks  = { path = "../vlm-tasks" }   # NEW
```

- [ ] **Step 2: Delete qwen/src/task.rs**

```bash
rm /Users/user/Develop/findit-studio/qwen/src/task.rs
```

- [ ] **Step 3: Update qwen/src/lib.rs to re-export from vlm-tasks**

In `qwen/src/lib.rs`:
- Remove `pub mod task;`
- Remove `pub use crate::task::{ParseError, Task};` from the prelude
- Add `pub use vlm_tasks::{ParseError, SceneAnalysis, Task};`

(Keep `pub mod engine;`, `pub mod error;`, `pub mod scene;`, and all `pub use crate::engine::*` / `pub use crate::error::*` re-exports.)

- [ ] **Step 4: Update qwen/src/scene.rs to use vlm-tasks SceneAnalysis**

In `qwen/src/scene.rs`:
- Replace the `pub struct SceneAnalysis { ... }` definition and the entire `impl SceneAnalysis { ... }` block (lines ~53-275) with `pub use vlm_tasks::SceneAnalysis;`
- Update `impl Task for SceneTask { type Output = SceneAnalysis; ... }` — should already work since `SceneAnalysis` is the same type, but `Task` is now `vlm_tasks::Task`. Add `use vlm_tasks::Task;` at the top.
- Inside `parse()`, the `into_scene_analysis()` helper builds the value using `SceneAnalysis::new().with_scene(...)...` — those builder methods come from vlm-tasks, no change needed.

- [ ] **Step 5: Run qwen unit tests**

```bash
cd /Users/user/Develop/findit-studio/qwen
cargo test --lib
```

Expected: all existing tests pass. The parser tests in `scene.rs` should be unchanged in behavior.

- [ ] **Step 6: Run clippy on qwen**

```bash
cd /Users/user/Develop/findit-studio/qwen
cargo clippy --lib --all-targets -- -D warnings
```

Expected: no new warnings.

- [ ] **Step 7: Verify integration test still compiles**

```bash
cd /Users/user/Develop/findit-studio/qwen
cargo build --features integration --tests
```

Expected: clean build. (Don't need to run — requires `QWEN_MODEL_PATH`.)

- [ ] **Step 8: Commit**

```bash
cd /Users/user/Develop/findit-studio/qwen
git add Cargo.toml Cargo.lock src/lib.rs src/scene.rs
git rm src/task.rs
git commit -m "$(cat <<'EOF'
Migrate to vlm-tasks for shared Task trait + SceneAnalysis

Three types move out of qwen into the new vlm-tasks crate:
- Task trait → vlm_tasks::Task (re-exported as qwen::Task)
- ParseError → vlm_tasks::ParseError (re-exported as qwen::ParseError)
- SceneAnalysis → vlm_tasks::SceneAnalysis (re-exported as qwen::SceneAnalysis)

qwen::scene::SceneTask stays here — the prompt, schema, and parser
fallback are qwen-tuned (different from what lfm will need for the
450M model). The SceneAnalysis values they produce are interchangeable
between qwen and lfm consumers.

Behavior unchanged. All existing tests pass without modification.
External imports continue to work via re-exports.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: lfm crate

### Task 3: Replace template-rs with lfm scaffold

**Files:**
- Modify: `lfm/Cargo.toml` (replace template-rs)
- Create: `lfm/src/` directory tree
- Move: `lfm/src/lib.rs` (overwrite the template stub)
- Create: `lfm/models/` with the 3 small files copied from `.lfm-model-cache/model/`
- Modify: `lfm/.gitignore` if needed

- [ ] **Step 1: Rewrite Cargo.toml**

Overwrite `lfm/Cargo.toml`:

```toml
[package]
name         = "lfm"
version      = "0.1.0"
edition      = "2024"
rust-version = "1.95"
description  = "Rust ONNX inference for LiquidAI LFM2.5-VL (vision-language) models"
license      = "MIT OR Apache-2.0"
include      = [
  "src/**/*.rs",
  "examples/**/*.rs",
  "benches/**/*.rs",
  "models/**",
  "build.rs",
  "Cargo.toml",
  "README.md",
  "CHANGELOG.md",
  "LICENSE-*",
]

[dependencies]
vlm-tasks   = { path = "../vlm-tasks" }
qwen        = { path = "../qwen", optional = true }   # only for examples/qwen_compare.rs (feature = "comparison")
ort         = { version = "2.0.0-rc.12", optional = true }
tokenizers  = { version = "0.23", optional = true }
llguidance  = { version = "1.7", optional = true }
minijinja   = { version = "2", optional = true, default-features = false, features = ["builtins"] }
image       = { version = "0.25", default-features = false }
smol_str    = "0.3"
thiserror   = "2"
tracing     = "0.1"
serde       = { version = "1", features = ["derive"], optional = true }
serde_json  = { version = "1", optional = true }

[dev-dependencies]
serde_json = "1"

[target.'cfg(not(target_arch = "wasm32"))'.dev-dependencies]
criterion = "0.8"

[features]
default     = ["inference", "bundled", "decoders"]
inference   = ["dep:ort", "dep:tokenizers", "dep:llguidance", "dep:minijinja"]
bundled     = ["inference"]
decoders    = ["image/jpeg", "image/png"]
serde       = ["dep:serde", "dep:serde_json", "smol_str/serde", "vlm-tasks/serde"]
cuda        = ["inference", "ort/cuda"]
tensorrt    = ["inference", "ort/tensorrt"]
directml    = ["inference", "ort/directml"]
rocm        = ["inference", "ort/rocm"]
coreml      = ["inference", "ort/coreml"]
integration = ["inference"]
# Opt-in cross-engine comparison example (loads BOTH qwen and lfm).
# Pulls qwen as a path dep — heavy (mistralrs + Metal). Off by default.
comparison  = ["inference", "dep:qwen"]

[[test]]
name              = "integration"
path              = "tests/integration.rs"
required-features = ["integration"]

[[example]]
name              = "smoke"
required-features = ["inference"]
[[example]]
name              = "scene_analysis"
required-features = ["inference"]
[[example]]
name              = "qwen_compare"
required-features = ["comparison"]
[[example]]
name              = "preprocess_only"
# wasm-compat showcase — no `inference` required.

[[bench]]
name    = "bench_preproc"
harness = false
[[bench]]
name    = "bench_tile_grid"
harness = false
[[bench]]
name    = "bench_chat_template"
harness = false
required-features = ["inference"]

[profile.bench]
opt-level         = 3
debug             = false
codegen-units     = 1
lto               = 'thin'
incremental       = false
debug-assertions  = false
overflow-checks   = false
rpath             = false

[package.metadata.docs.rs]
features     = ["inference", "bundled", "decoders", "serde"]
rustdoc-args = ["--cfg", "docsrs"]

[lints.rust]
rust_2018_idioms     = "warn"
single_use_lifetimes = "warn"
unexpected_cfgs      = { level = "warn", check-cfg = ['cfg(docsrs)'] }
```

- [ ] **Step 2: Create directory structure + delete template stubs**

```bash
cd /Users/user/Develop/findit-studio/lfm
mkdir -p src/preproc src/runtime models
rm src/lib.rs benches/foo.rs tests/foo.rs examples/foo.rs
rmdir benches tests/fixtures 2>/dev/null || true   # only if empty
mkdir -p tests/fixtures benches examples
```

- [ ] **Step 3: Copy bundled model files**

```bash
cd /Users/user/Develop/findit-studio/lfm
cp /Users/user/Develop/findit-studio/.lfm-model-cache/model/tokenizer.json        models/
cp /Users/user/Develop/findit-studio/.lfm-model-cache/model/chat_template.jinja   models/
cp /Users/user/Develop/findit-studio/.lfm-model-cache/model/preprocessor_config.json models/
ls -lh models/
```

Expected: `tokenizer.json` 4.5 MB, `chat_template.jinja` 3.8 KB, `preprocessor_config.json` 0.7 KB.

- [ ] **Step 4: Create LICENSE files**

```bash
cd /Users/user/Develop/findit-studio/lfm
# Already exist from template — verify they're MIT and Apache-2.0
head -3 LICENSE-MIT LICENSE-APACHE
```

If missing, copy from `qwen/`.

- [ ] **Step 5: Write skeleton lib.rs (will be expanded later)**

Create `lfm/src/lib.rs`:

```rust
//! Rust ONNX inference for LiquidAI LFM2.5-VL (vision-language) models.
//!
//! See `docs/superpowers/specs/2026-05-03-lfm-vlm-wrapper-design.md`
//! for the full design rationale.
//!
//! ## Model weights license
//!
//! This crate is dual-licensed under MIT OR Apache-2.0. **The model
//! weights it wraps are NOT** — LFM2.5-VL-450M ships under the LFM
//! Open License v1.0 (`lfm1.0`, see <https://www.liquid.ai/lfm-license>).
//! Verify your use case complies with Liquid AI's terms separately
//! from this crate's license.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes, missing_docs)]

pub mod error;
// More modules added in subsequent tasks.

pub use error::{Error, Result};
```

- [ ] **Step 6: Verify the scaffold compiles (with a stub error.rs)**

Create a minimal stub `lfm/src/error.rs` (will be expanded in Task 4):

```rust
//! Error type for the lfm crate. (Stub — full implementation in Task 4.)

pub type Result<T> = std::result::Result<T, Error>;

/// Crate-level error type.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
  /// Placeholder until Task 4 fills in the variants.
  #[error("not yet implemented")]
  NotYetImplemented,
}
```

- [ ] **Step 7: Run the build**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo build --no-default-features
```

Expected: clean build (no inference features pulled in).

- [ ] **Step 8: Commit**

```bash
cd /Users/user/Develop/findit-studio/lfm
git add Cargo.toml Cargo.lock src/lib.rs src/error.rs models/ .gitignore
git rm benches/foo.rs tests/foo.rs examples/foo.rs 2>/dev/null || true
git commit -m "$(cat <<'EOF'
Replace template-rs scaffold with lfm crate

Cargo.toml rewritten with:
- Real package name + description + license + include list
- Dependencies: vlm-tasks (path), ort 2.0.0-rc.12, tokenizers 0.23,
  llguidance 1.7, minijinja 2, image 0.25, smol_str, thiserror, tracing
- Features: inference (default), bundled (default), decoders (default),
  serde, integration, cuda/tensorrt/directml/rocm/coreml EP gates
- Three test/example/bench targets per the spec

Model files bundled into models/:
- tokenizer.json (4.5 MB, behind `bundled` feature)
- chat_template.jinja (3.8 KB, always shipped)
- preprocessor_config.json (0.7 KB, build-fixture only)

src/error.rs is a stub; lib.rs has only the error re-export. Subsequent
tasks fill in error.rs, then options/preproc/runtime/engine/etc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: error.rs

**Files:**
- Modify: `lfm/src/error.rs` (replace stub with full enum)

- [ ] **Step 1: Write the failing tests**

Append to `lfm/src/error.rs`:

```rust
#[cfg(test)]
mod tests {
  use super::*;
  use smol_str::SmolStr;

  #[test]
  fn ll_guidance_dead_end_state_inlined_smolstr() {
    let e = Error::LlGuidanceDeadEnd { step: 5, state: SmolStr::new_inline("regex stuck") };
    let msg = format!("{e}");
    assert!(msg.contains("step 5"));
    assert!(msg.contains("regex stuck"));
  }

  #[test]
  fn invalid_request_uses_static_str() {
    fn classify(e: &Error) -> &'static str {
      match e { Error::InvalidRequest(s) => s, _ => "other" }
    }
    let e = Error::InvalidRequest("max_new_tokens must be > 0");
    assert_eq!(classify(&e), "max_new_tokens must be > 0");
  }

  // These two tests use Error::tokenizer / Error::llguidance — both
  // gated on `feature = "inference"`, so the tests must be too.
  // Without these gates, `cargo test --no-default-features error::`
  // fails to compile.
  #[cfg(feature = "inference")]
  #[test]
  fn tokenizer_constructor_round_trips_through_box_dyn() {
    let inner = std::io::Error::new(std::io::ErrorKind::InvalidData, "bad utf8");
    let e = Error::tokenizer(inner);
    let display = format!("{e}");
    assert!(display.contains("bad utf8"));
  }

  #[cfg(feature = "inference")]
  #[test]
  fn llguidance_constructor_round_trips_through_box_dyn() {
    let inner = std::io::Error::new(std::io::ErrorKind::Other, "matcher exhausted");
    let e = Error::llguidance(inner);
    let display = format!("{e}");
    assert!(display.contains("matcher exhausted"));
  }

  #[test]
  fn from_io_works_via_question_mark() {
    fn inner() -> Result<()> {
      std::fs::read("/no/such/path/here").map(|_| ())?;
      Ok(())
    }
    let e = inner().unwrap_err();
    assert!(matches!(e, Error::Io(_)));
  }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo test --lib --no-default-features error::
```

Expected: FAIL — `Error` enum has only `NotYetImplemented`; the test variants don't exist.

- [ ] **Step 3: Replace stub with full Error enum**

Overwrite `lfm/src/error.rs`:

```rust
//! Error type for the `lfm` crate.
//!
//! Single `Error` enum (matches siglip2/egemma idiom). Style rules:
//! 1. Wrap, don't stringify external errors (use `Box<dyn Error + Send + Sync>`).
//! 2. `SmolStr` for runtime-built short strings.
//! 3. `&'static str` for fixed literals (outlet names, `InvalidRequest` reasons).
//! 4. `#[error(transparent)]` for already-self-describing wrapped errors.
//! 5. Named constructors when `From` would conflict (`Error::tokenizer`, `Error::llguidance`).

use std::path::PathBuf;
use smol_str::SmolStr;
use thiserror::Error;

/// Crate-level Result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Crate-level Error enum.
///
/// `#[non_exhaustive]` so adding new variants in a future minor release
/// (notably the streaming + tool-calling variants planned in §10 of the
/// design spec) is not a SemVer break — downstream `match` arms must
/// include a wildcard.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
  // ===== Loading =====

  /// File not found at the given path.
  #[error("file not found: {0}")]
  NotFound(PathBuf),

  /// I/O error.
  #[error(transparent)]
  Io(#[from] std::io::Error),

  /// `ort` session-build or session-run failure.
  #[cfg(feature = "inference")]
  #[error(transparent)]
  Ort(#[from] ort::Error),

  /// `tokenizers` load/use failure. Boxed to preserve the source chain.
  #[cfg(feature = "inference")]
  #[error(transparent)]
  Tokenizer(Box<dyn std::error::Error + Send + Sync>),

  // ===== Session validation =====

  /// ONNX session input/output dtype or presence mismatch.
  #[cfg(feature = "inference")]
  #[error("session contract mismatch on {input}: expected {expected}, got {got:?}")]
  SessionContractMismatch {
    input: &'static str,
    expected: &'static str,
    got: ort::value::TensorElementType,
  },

  /// ONNX session shape mismatch.
  #[cfg(feature = "inference")]
  #[error("session shape mismatch on {input}: expected {expected}, got {got:?}")]
  SessionShapeMismatch {
    input: &'static str,
    expected: &'static str,
    got: Vec<i64>,
  },

  /// Decoder cache layer count or sparse-index mismatch.
  #[cfg(feature = "inference")]
  #[error(
    "decoder cache mismatch: expected {expected_conv} conv + {expected_attn} attn, \
     got {got_conv} conv + {got_attn} attn"
  )]
  DecoderCacheMismatch {
    expected_conv: usize,
    expected_attn: usize,
    got_conv: usize,
    got_attn: usize,
  },

  // ===== Preprocessing =====

  /// Image decode failure.
  #[cfg(feature = "decoders")]
  #[error(transparent)]
  ImageDecode(#[from] image::ImageError),

  /// Image too small for the current `ImageBudget`.
  #[error("image {w}x{h} too small for ImageBudget (need at least {min_w}x{min_h})")]
  ImageTooSmall { w: u32, h: u32, min_w: u32, min_h: u32 },

  /// Tile-grid algorithm could not satisfy the budget.
  #[error("no valid tile grid for image {w}x{h}")]
  TileGridImpossible { w: u32, h: u32 },

  // ===== Tokenization / template =====

  /// `<image>` placeholder count mismatch with input image count.
  #[error("expected {expected} <image> placeholder(s) in prompt, got {got}")]
  ImageTokenCountMismatch { expected: usize, got: usize },

  // ===== Generation =====

  /// llguidance compile or matcher failure.
  #[cfg(feature = "inference")]
  #[error(transparent)]
  LlGuidance(Box<dyn std::error::Error + Send + Sync>),

  /// llguidance produced an all-zero next-token mask.
  #[error("llguidance produced empty mask at step {step}: {state}")]
  LlGuidanceDeadEnd { step: usize, state: SmolStr },

  /// Generation hit `max_new_tokens` without EOS or schema-complete.
  #[error("hit max_new_tokens={max} (schema_complete={schema_complete})")]
  MaxTokensExceeded { max: usize, schema_complete: bool },

  /// Detokenize produced invalid UTF-8.
  #[error("detokenize produced invalid UTF-8")]
  InvalidUtf8,

  /// Generation produced no output.
  #[error("generation produced empty output")]
  Empty,

  // ===== Configuration =====

  /// Invalid `RequestOptions`.
  #[error("invalid RequestOptions: {0}")]
  InvalidRequest(&'static str),

  /// Invalid `ImageBudget`.
  #[error("invalid ImageBudget: {0}")]
  InvalidBudget(&'static str),

  // ===== Task parse =====

  /// Forwarded from `vlm_tasks::ParseError`.
  #[error(transparent)]
  Parse(#[from] vlm_tasks::ParseError),
}

impl Error {
  /// Wrap any `Error + Send + Sync` source as a `Tokenizer` variant.
  #[cfg(feature = "inference")]
  pub(crate) fn tokenizer<E>(e: E) -> Self
  where E: Into<Box<dyn std::error::Error + Send + Sync>> {
    Self::Tokenizer(e.into())
  }

  /// Wrap any `Error + Send + Sync` source as an `LlGuidance` variant.
  #[cfg(feature = "inference")]
  pub(crate) fn llguidance<E>(e: E) -> Self
  where E: Into<Box<dyn std::error::Error + Send + Sync>> {
    Self::LlGuidance(e.into())
  }
}

#[cfg(test)]
mod tests {
  // (tests from Step 1 — already in place)
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo test --lib --no-default-features error::
cargo test --lib error::
```

Expected: all 5 tests pass under both feature configs.

- [ ] **Step 5: Run clippy**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo clippy --lib --all-targets --no-default-features -- -D warnings
cargo clippy --lib --all-targets -- -D warnings
```

Expected: no warnings.

- [ ] **Step 6: Commit**

```bash
cd /Users/user/Develop/findit-studio/lfm
git add src/error.rs
git commit -m "feat(error): full Error enum with boxed sources + named constructors

Replaces the NotYetImplemented stub with the full taxonomy from spec §9:
- Loading: NotFound, Io, Ort, Tokenizer
- Session validation: SessionContractMismatch, SessionShapeMismatch, DecoderCacheMismatch
- Preprocessing: ImageDecode, ImageTooSmall, TileGridImpossible
- Tokenization: ImageTokenCountMismatch
- Generation: LlGuidance, LlGuidanceDeadEnd, MaxTokensExceeded, InvalidUtf8, Empty
- Configuration: InvalidRequest, InvalidBudget
- Task parse: Parse (forwarded from vlm_tasks::ParseError)

Style: #[non_exhaustive], boxed external sources (Tokenizer + LlGuidance),
SmolStr for LlGuidanceDeadEnd::state, &'static str for fixed reasons,
#[error(transparent)] for self-describing wrapped errors. Named
constructors Error::tokenizer / Error::llguidance disambiguate the
two Box<dyn> variants.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 5: options.rs

**Files:**
- Create: `lfm/src/options.rs`
- Modify: `lfm/src/lib.rs` (add `pub mod options;` + re-exports)

- [ ] **Step 1: Write the failing tests**

Create `lfm/src/options.rs` with the test module first:

```rust
//! Configuration types: `RequestOptions`, `ImageBudget`, `ThreadOptions`, `Options`.

#[cfg(feature = "inference")]
pub use ort::session::builder::GraphOptimizationLevel;

use crate::error::{Error, Result};

// (impls follow in Step 3)

#[cfg(test)]
mod tests {
  use super::*;

  // ===== RequestOptions =====

  #[test]
  fn request_options_new_matches_model_card() {
    let r = RequestOptions::new();
    assert_eq!(r.temperature(), 0.1);
    assert_eq!(r.min_p(), 0.15);
    assert_eq!(r.repetition_penalty(), 1.05);
    assert_eq!(r.max_new_tokens(), 512);
  }

  #[test]
  fn request_options_deterministic_is_greedy() {
    let r = RequestOptions::deterministic();
    assert_eq!(r.temperature(), 0.0);
    assert_eq!(r.repetition_penalty(), 1.05);
  }

  #[test]
  fn request_options_validate_rejects_bad_inputs() {
    assert!(RequestOptions::new().with_max_new_tokens(0).validate().is_err());
    assert!(RequestOptions::new().with_temperature(-1.0).validate().is_err());
    assert!(RequestOptions::new().with_min_p(2.0).validate().is_err());
    assert!(RequestOptions::new().with_repetition_penalty(0.5).validate().is_err());
  }

  #[test]
  fn request_options_with_chains() {
    let r = RequestOptions::new()
      .with_temperature(0.3)
      .with_min_p(0.05)
      .with_repetition_penalty(1.10)
      .with_max_new_tokens(1024);
    assert_eq!(r.temperature(), 0.3);
    assert_eq!(r.max_new_tokens(), 1024);
  }

  // ===== ImageBudget =====

  #[test]
  fn image_budget_new_matches_preprocessor_config() {
    let b = ImageBudget::new();
    assert_eq!(b.min_image_tokens(), 64);
    assert_eq!(b.max_image_tokens(), 256);
    assert_eq!(b.min_tiles(), 2);
    assert_eq!(b.max_tiles(), 10);
    assert!(b.use_thumbnail());
  }

  #[test]
  fn image_budget_fast_is_smaller() {
    let f = ImageBudget::fast();
    assert!(f.max_image_tokens() < ImageBudget::new().max_image_tokens());
    assert!(!f.use_thumbnail());
  }

  #[test]
  fn image_budget_validate_rejects_bad_inputs() {
    let mut b = ImageBudget::new();
    b.set_min_image_tokens(0);
    assert!(b.validate().is_err());
    let mut b2 = ImageBudget::new();
    b2.set_max_image_tokens(b2.min_image_tokens() - 1);
    assert!(b2.validate().is_err());
  }

  // ===== Send/Sync =====

  #[test]
  fn options_are_send_sync_copy_or_clone() {
    fn req<T: Send + Sync>() {}
    req::<RequestOptions>();
    req::<ImageBudget>();
    req::<ThreadOptions>();
    req::<Options>();
  }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo test --lib --no-default-features options::
```

Expected: FAIL — types not defined.

- [ ] **Step 3: Implement the four option types**

Replace the file with:

```rust
//! Configuration types: `RequestOptions`, `ImageBudget`, `ThreadOptions`, `Options`.

use crate::error::{Error, Result};

#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use ort::session::builder::GraphOptimizationLevel;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// =========================================================================
// RequestOptions
// =========================================================================

/// Sampler configuration applied per call by `Engine::run` / `generate` /
/// their `_with` variants.
///
/// LFM2.5-VL uses **min_p sampling** (NOT top_p / top_k); fields reflect
/// the model card's recommended sampler. Two named presets ship out of
/// the box: `RequestOptions::new()` (model-card defaults) and
/// `RequestOptions::deterministic()` (greedy + retained repetition_penalty).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RequestOptions {
  temperature: f32,
  min_p: f32,
  repetition_penalty: f32,
  max_new_tokens: usize,
}

impl RequestOptions {
  /// Defaults from the LFM2.5-VL-450M model card text:
  /// `temperature=0.1`, `min_p=0.15`, `repetition_penalty=1.05`,
  /// `max_new_tokens=512`. Best output quality; not bit-stable.
  ///
  /// Source: <https://huggingface.co/LiquidAI/LFM2.5-VL-450M> §"Inference".
  pub const fn new() -> Self {
    Self { temperature: 0.1, min_p: 0.15, repetition_penalty: 1.05, max_new_tokens: 512 }
  }

  /// Indexing-safe greedy: `temperature=0.0`, `repetition_penalty=1.05`
  /// retained (greedy without it loops on small models). `min_p` is
  /// irrelevant under argmax.
  ///
  /// **Bit-stability caveat:** greedy is necessary but not sufficient.
  /// ORT bit-stability also requires `intra_threads=1`, `inter_threads=1`,
  /// and CPU-only EP. See `ThreadOptions` + EP feature flags.
  pub const fn deterministic() -> Self {
    Self { temperature: 0.0, min_p: 0.0, repetition_penalty: 1.05, max_new_tokens: 512 }
  }

  pub const fn temperature(&self) -> f32 { self.temperature }
  pub const fn min_p(&self) -> f32 { self.min_p }
  pub const fn repetition_penalty(&self) -> f32 { self.repetition_penalty }
  pub const fn max_new_tokens(&self) -> usize { self.max_new_tokens }

  pub const fn with_temperature(mut self, v: f32) -> Self { self.temperature = v; self }
  pub const fn with_min_p(mut self, v: f32) -> Self { self.min_p = v; self }
  pub const fn with_repetition_penalty(mut self, v: f32) -> Self { self.repetition_penalty = v; self }
  pub const fn with_max_new_tokens(mut self, v: usize) -> Self { self.max_new_tokens = v; self }

  pub fn set_temperature(&mut self, v: f32) -> &mut Self { self.temperature = v; self }
  pub fn set_min_p(&mut self, v: f32) -> &mut Self { self.min_p = v; self }
  pub fn set_repetition_penalty(&mut self, v: f32) -> &mut Self { self.repetition_penalty = v; self }
  pub fn set_max_new_tokens(&mut self, v: usize) -> &mut Self { self.max_new_tokens = v; self }

  /// Validate per spec §13.2 #19. Returns `Error::InvalidRequest(reason)` on failure.
  pub const fn validate(&self) -> Result<()> {
    if self.temperature < 0.0 { return Err(Error::InvalidRequest("temperature must be >= 0.0")); }
    if self.min_p < 0.0 || self.min_p > 1.0 { return Err(Error::InvalidRequest("min_p must be in [0.0, 1.0]")); }
    if self.repetition_penalty < 1.0 { return Err(Error::InvalidRequest("repetition_penalty must be >= 1.0")); }
    if self.max_new_tokens == 0 { return Err(Error::InvalidRequest("max_new_tokens must be > 0")); }
    Ok(())
  }
}

impl Default for RequestOptions {
  fn default() -> Self { Self::new() }
}

// =========================================================================
// ImageBudget
// =========================================================================

/// Per-image preprocessing budget. Note: `max_image_tokens` is **asymmetric
/// across paths** — it bounds the single-tile path's `smart_resize` and
/// the thumbnail's `smart_resize`, but does NOT bound the multi-tile
/// path's main-tile total (which is `rows × cols × 256`, capped only by
/// `max_tiles`). See spec §13.3 #14 for the full discussion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageBudget {
  min_image_tokens: usize,
  max_image_tokens: usize,
  min_tiles: usize,
  max_tiles: usize,
  use_thumbnail: bool,
  max_pixels_tolerance_x100: u32,  // store as integer to keep Eq; divide by 100 to use
}

impl ImageBudget {
  /// `preprocessor_config.json` defaults: min=64 tokens, max=256 tokens,
  /// min=2 tiles, max=10 tiles, thumbnail on, max_pixels_tolerance=2.0.
  pub const fn new() -> Self {
    Self {
      min_image_tokens: 64, max_image_tokens: 256,
      min_tiles: 2, max_tiles: 10,
      use_thumbnail: true,
      max_pixels_tolerance_x100: 200,
    }
  }

  /// Speed-optimized: `max_image_tokens=64`, `max_tiles=4`, no thumbnail.
  /// ~3-4× speedup at lower per-frame quality.
  pub const fn fast() -> Self {
    Self {
      min_image_tokens: 32, max_image_tokens: 64,
      min_tiles: 2, max_tiles: 4,
      use_thumbnail: false,
      max_pixels_tolerance_x100: 200,
    }
  }

  /// Quality-optimized — currently identical to `new()`; kept as a
  /// named preset so future config changes don't silently re-tune the
  /// "I want best quality" call site.
  pub const fn quality() -> Self { Self::new() }

  pub const fn min_image_tokens(&self) -> usize { self.min_image_tokens }
  pub const fn max_image_tokens(&self) -> usize { self.max_image_tokens }
  pub const fn min_tiles(&self) -> usize { self.min_tiles }
  pub const fn max_tiles(&self) -> usize { self.max_tiles }
  pub const fn use_thumbnail(&self) -> bool { self.use_thumbnail }
  pub fn max_pixels_tolerance(&self) -> f32 { self.max_pixels_tolerance_x100 as f32 / 100.0 }

  pub const fn with_min_image_tokens(mut self, v: usize) -> Self { self.min_image_tokens = v; self }
  pub const fn with_max_image_tokens(mut self, v: usize) -> Self { self.max_image_tokens = v; self }
  pub const fn with_min_tiles(mut self, v: usize) -> Self { self.min_tiles = v; self }
  pub const fn with_max_tiles(mut self, v: usize) -> Self { self.max_tiles = v; self }
  pub const fn with_use_thumbnail(mut self, v: bool) -> Self { self.use_thumbnail = v; self }

  pub fn set_min_image_tokens(&mut self, v: usize) -> &mut Self { self.min_image_tokens = v; self }
  pub fn set_max_image_tokens(&mut self, v: usize) -> &mut Self { self.max_image_tokens = v; self }
  pub fn set_min_tiles(&mut self, v: usize) -> &mut Self { self.min_tiles = v; self }
  pub fn set_max_tiles(&mut self, v: usize) -> &mut Self { self.max_tiles = v; self }
  pub fn set_use_thumbnail(&mut self, v: bool) -> &mut Self { self.use_thumbnail = v; self }

  /// Validate per spec §13.2 #19.
  pub const fn validate(&self) -> Result<()> {
    if self.min_image_tokens == 0 { return Err(Error::InvalidBudget("min_image_tokens must be > 0")); }
    if self.max_image_tokens < self.min_image_tokens { return Err(Error::InvalidBudget("max_image_tokens must be >= min_image_tokens")); }
    if self.min_tiles == 0 { return Err(Error::InvalidBudget("min_tiles must be > 0")); }
    if self.max_tiles < self.min_tiles { return Err(Error::InvalidBudget("max_tiles must be >= min_tiles")); }
    if self.max_pixels_tolerance_x100 == 0 { return Err(Error::InvalidBudget("max_pixels_tolerance must be > 0.0")); }
    Ok(())
  }
}

impl Default for ImageBudget {
  fn default() -> Self { Self::new() }
}

// =========================================================================
// ThreadOptions
// =========================================================================

/// ORT thread configuration. Mirrors siglip2/egemma `ThreadOptions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ThreadOptions {
  intra_threads: Option<usize>,
  inter_threads: Option<usize>,
}

impl ThreadOptions {
  /// `None`/`None` = let ort pick defaults.
  pub const fn new() -> Self { Self { intra_threads: None, inter_threads: None } }

  /// Indexing-safe single-threaded — pair with `RequestOptions::deterministic()`
  /// for end-to-end bit-stability.
  pub const fn deterministic() -> Self { Self { intra_threads: Some(1), inter_threads: Some(1) } }

  pub const fn intra_threads(&self) -> Option<usize> { self.intra_threads }
  pub const fn inter_threads(&self) -> Option<usize> { self.inter_threads }
  pub const fn with_intra_threads(mut self, v: usize) -> Self { self.intra_threads = Some(v); self }
  pub const fn with_inter_threads(mut self, v: usize) -> Self { self.inter_threads = Some(v); self }
}

impl Default for ThreadOptions {
  fn default() -> Self { Self::new() }
}

// =========================================================================
// Options (top-level)
// =========================================================================

/// Top-level engine configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Options {
  request: RequestOptions,
  image_budget: ImageBudget,
  thread: ThreadOptions,
  #[cfg(feature = "inference")]
  optimization_level: GraphOptLevelMirror,
}

impl Options {
  /// Defaults: `RequestOptions::deterministic()`, `ImageBudget::new()`,
  /// `ThreadOptions::default()`, `GraphOptimizationLevel::Level1`
  /// (matches siglip2/egemma — higher levels can subtly alter numerics).
  pub const fn new() -> Self {
    Self {
      request: RequestOptions::deterministic(),
      image_budget: ImageBudget::new(),
      thread: ThreadOptions::new(),
      #[cfg(feature = "inference")]
      optimization_level: GraphOptLevelMirror::Level1,
    }
  }

  pub const fn request(&self) -> &RequestOptions { &self.request }
  pub const fn image_budget(&self) -> &ImageBudget { &self.image_budget }
  pub const fn thread(&self) -> &ThreadOptions { &self.thread }
  #[cfg(feature = "inference")]
  pub fn optimization_level(&self) -> GraphOptimizationLevel {
    self.optimization_level.into()
  }

  pub const fn with_request(mut self, r: RequestOptions) -> Self { self.request = r; self }
  pub const fn with_image_budget(mut self, b: ImageBudget) -> Self { self.image_budget = b; self }
  pub const fn with_thread(mut self, t: ThreadOptions) -> Self { self.thread = t; self }
  #[cfg(feature = "inference")]
  pub fn with_optimization_level(mut self, lvl: GraphOptimizationLevel) -> Self {
    self.optimization_level = lvl.into();
    self
  }
}

impl Default for Options {
  fn default() -> Self { Self::new() }
}

/// Serde-friendly mirror of `GraphOptimizationLevel` (which doesn't
/// derive Serialize/Deserialize directly). Mirrors siglip2 pattern.
#[cfg(feature = "inference")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
enum GraphOptLevelMirror { Disable, Level1, Level2, Level3, All }

#[cfg(feature = "inference")]
impl From<GraphOptimizationLevel> for GraphOptLevelMirror {
  fn from(v: GraphOptimizationLevel) -> Self {
    match v {
      GraphOptimizationLevel::Disable => Self::Disable,
      GraphOptimizationLevel::Level1 => Self::Level1,
      GraphOptimizationLevel::Level2 => Self::Level2,
      GraphOptimizationLevel::Level3 => Self::Level3,
      GraphOptimizationLevel::All => Self::All,
    }
  }
}

#[cfg(feature = "inference")]
impl From<GraphOptLevelMirror> for GraphOptimizationLevel {
  fn from(v: GraphOptLevelMirror) -> Self {
    match v {
      GraphOptLevelMirror::Disable => Self::Disable,
      GraphOptLevelMirror::Level1 => Self::Level1,
      GraphOptLevelMirror::Level2 => Self::Level2,
      GraphOptLevelMirror::Level3 => Self::Level3,
      GraphOptLevelMirror::All => Self::All,
    }
  }
}

// (test module from Step 1)
```

Add to `lfm/src/lib.rs`:

```rust
pub mod options;
pub use options::{ImageBudget, Options, RequestOptions, ThreadOptions};
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use options::GraphOptimizationLevel;
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo test --lib --no-default-features options::
cargo test --lib options::
```

Expected: all 8 tests pass under both feature configs.

- [ ] **Step 5: Run clippy**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo clippy --lib --all-targets --no-default-features -- -D warnings
cargo clippy --lib --all-targets -- -D warnings
```

- [ ] **Step 6: Commit**

```bash
cd /Users/user/Develop/findit-studio/lfm
git add src/options.rs src/lib.rs
git commit -m "feat(options): RequestOptions + ImageBudget + ThreadOptions + Options

Per spec §6.3:
- RequestOptions: temperature, min_p, repetition_penalty, max_new_tokens.
  Two presets: ::new() (model-card 0.1/0.15/1.05/512), ::deterministic()
  (greedy 0.0/-/1.05/512). Validation per §13.2 #19.
- ImageBudget: min/max image tokens, min/max tiles, use_thumbnail,
  max_pixels_tolerance. Three presets: ::new(), ::fast(), ::quality().
  max_image_tokens is asymmetric across paths (see §13.3 #14 docstring).
- ThreadOptions: intra/inter threads (None = ort defaults).
  ::deterministic() pins both to 1 for bit-stable reproducibility.
- Options: top-level config. Default optimization_level = Level1
  (sibling-crate parity; higher levels can alter numerics under greedy).
- GraphOptLevelMirror: serde-friendly mirror enum for serialization
  (GraphOptimizationLevel itself doesn't derive serde).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 6: chat_template.rs

**Files:**
- Create: `lfm/src/chat_template.rs`
- Create: `lfm/tests/fixtures/chat_template_cases.json` (capture from upstream Jinja)
- Create: `lfm/tests/fixtures/image_expansion_cases.json` (capture from upstream)
- Modify: `lfm/src/lib.rs` (add `pub mod chat_template;` + re-exports)

This task uses **`minijinja`** for the heavy lifting (per spec §13.3 #12 trade-off — re-implementing the upstream Jinja template by hand is 250-400 LoC with subtle correctness risks; minijinja is well-tested and the `generation` tag we no-op).

- [ ] **Step 1: Capture chat-template fixtures from upstream Jinja**

Write a one-off Python helper that uses `transformers.AutoTokenizer` (already installed during Phase 0) to render 8 representative messages and dump the input → expected-output pairs:

```bash
cd /Users/user/Develop/findit-studio/lfm/tests/fixtures
python3 - <<'PY' > chat_template_cases.json
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-VL-450M", trust_remote_code=True)
cases = [
  {"name": "single_user_text", "messages": [{"role": "user", "content": "Hi"}]},
  {"name": "single_user_image_text", "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]},
  {"name": "system_user", "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}]},
  {"name": "user_assistant_user", "messages": [
    {"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"}, {"role": "user", "content": "2+2?"},
  ]},
  {"name": "multi_image_user", "messages": [{"role": "user", "content": [
    {"type": "image"}, {"type": "image"}, {"type": "text", "text": "Compare."},
  ]}]},
  {"name": "thinking_assistant", "messages": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "thinking": "let me think", "content": "A"},
    {"role": "user", "content": "Follow-up"},
  ]},
  {"name": "with_tools", "messages": [{"role": "user", "content": "Weather Paris?"}], "tools": [
    {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}},
  ]},
  {"name": "no_generation_prompt", "messages": [{"role": "user", "content": "Hi"}], "add_generation_prompt": False},
]
for c in cases:
  c["expected"] = tok.apply_chat_template(
    c["messages"], tools=c.get("tools"), add_generation_prompt=c.get("add_generation_prompt", True), tokenize=False,
  )
print(json.dumps(cases, indent=2))
PY
```

(If transformers v4.57 chokes on `TokenizersBackend` again here, fall back: render with `minijinja` against `models/chat_template.jinja` directly using the same input dicts. The point is to fix `expected` values that the Rust port must match exactly.)

- [ ] **Step 2: Write the failing tests**

Create `lfm/src/chat_template.rs`:

```rust
//! Chat-template rendering for LFM2.5-VL.
//!
//! Two-step pipeline:
//! 1. `apply_chat_template` renders the upstream Jinja template via
//!    `minijinja`, producing a chat-formatted prompt with literal
//!    `<image>` placeholders (one per image content item).
//! 2. `expand_image_placeholders` walks those placeholders and
//!    substitutes the per-image structure (`<|image_start|>` +
//!    per-tile `<|img_row_R_col_C|>` + `<image>` × tokens_per_tile +
//!    optional thumbnail + `<|image_end|>`) using each image's grid
//!    layout from `PreprocessedImage`.
//!
//! Special token strings + IDs (per Phase 0 G-resolved tokenizer.json).

use smol_str::SmolStr;

// ===== Special token constants =====

pub const BOS: &str = "<|startoftext|>";
pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const PAD: &str = "<|pad|>";
pub const IMAGE_TOKEN: &str = "<image>";
pub const IMAGE_START: &str = "<|image_start|>";
pub const IMAGE_END: &str = "<|image_end|>";
pub const IMAGE_THUMBNAIL: &str = "<|img_thumbnail|>";
pub const TOOL_CALL_START: &str = "<|tool_call_start|>";
pub const TOOL_CALL_END: &str = "<|tool_call_end|>";

pub const BOS_TOKEN_ID: u32 = 1;
pub const EOS_TOKEN_ID: u32 = 7;
pub const PAD_TOKEN_ID: u32 = 0;
pub const IMAGE_TOKEN_ID: u32 = 396;

// ===== Public API =====

/// Bundled Jinja source. Always shipped (3.8 KB).
pub const BUNDLED_CHAT_TEMPLATE_JINJA: &str = include_str!("../models/chat_template.jinja");

#[cfg(feature = "inference")]
mod render {
  use super::*;
  use serde::Serialize;

  /// Top-level chat-template entry point. Renders the bundled Jinja
  /// template via `minijinja`. Returns the chat-formatted prompt with
  /// literal `<image>` placeholders for image content items.
  ///
  /// Callers handle image expansion separately via
  /// [`expand_image_placeholders`] once they know the per-image grid layout.
  pub fn apply_chat_template(
    messages: &[Message<'_>],
    tools: Option<&serde_json::Value>,
    add_generation_prompt: bool,
  ) -> crate::error::Result<String> {
    use minijinja::{Environment, Value};

    let mut env = Environment::new();
    env.add_function("strftime_now", |fmt: String| {
      // Upstream uses %Y-%m-%d. We bake today's date at template-render time.
      // (The template uses it only inside the tools-block prefix.)
      Ok::<_, minijinja::Error>(format!("{}", chrono_like::today_yyyymmdd(&fmt)))
    });
    let tmpl = env.template_from_str(BUNDLED_CHAT_TEMPLATE_JINJA)
      .map_err(|e| crate::error::Error::tokenizer(e))?;

    let ctx = Value::from_serialize(&RenderContext {
      bos_token: BOS,
      messages,
      tools,
      add_generation_prompt,
    });
    tmpl.render(ctx).map_err(|e| crate::error::Error::tokenizer(e))
  }

  #[derive(Serialize)]
  struct RenderContext<'a> {
    bos_token: &'a str,
    messages: &'a [Message<'a>],
    tools: Option<&'a serde_json::Value>,
    add_generation_prompt: bool,
  }

  /// Tiny stub for the `strftime_now` Jinja function — the upstream
  /// template uses it only for the tools-block prefix's `Today's date:`.
  /// Returns YYYY-MM-DD per upstream behavior.
  mod chrono_like {
    pub fn today_yyyymmdd(_fmt: &str) -> String {
      // No chrono dep in our Cargo.toml — implement manually via
      // SystemTime + a tiny date-from-epoch routine. (Inline below or
      // factor to a util module.)
      use std::time::{SystemTime, UNIX_EPOCH};
      let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
      // Days since epoch (1970-01-01).
      let days = secs / 86400;
      let (y, m, d) = days_to_ymd(days as i64);
      format!("{:04}-{:02}-{:02}", y, m, d)
    }
    /// Convert days-since-1970-01-01 to (year, month, day). Howard Hinnant's algo.
    fn days_to_ymd(days: i64) -> (i64, u32, u32) {
      let z = days + 719468;
      let era = if z >= 0 { z } else { z - 146096 } / 146097;
      let doe = (z - era * 146097) as u64;
      let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
      let y = yoe as i64 + era * 400;
      let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
      let mp = (5 * doy + 2) / 153;
      let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
      let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
      let y = if m <= 2 { y + 1 } else { y };
      (y, m, d)
    }
  }
}
#[cfg(feature = "inference")]
pub use render::apply_chat_template;

/// One message in a chat-template render call.
#[cfg(feature = "inference")]
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message<'a> {
  System { content: &'a str },
  User { content: UserContent<'a> },
  Assistant {
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<&'a str>,
  },
}

#[cfg(feature = "inference")]
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum UserContent<'a> {
  Text(&'a str),
  Multimodal(Vec<ContentItem<'a>>),
}

#[cfg(feature = "inference")]
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentItem<'a> {
  Image,
  Text { text: &'a str },
}

/// Walk a chat-formatted prompt and expand each literal `<image>`
/// placeholder into the per-image structure. The Engine calls this
/// AFTER `apply_chat_template` and BEFORE tokenization.
///
/// Returns `Error::ImageTokenCountMismatch` if the placeholder count
/// in `prompt` doesn't match `images.len()`.
pub fn expand_image_placeholders(
  prompt: &str,
  images: &[crate::preproc::PreprocessedImage],
) -> crate::error::Result<String> {
  // Split on the literal placeholder. After splitting on N occurrences
  // we get N+1 pieces; placeholders go between consecutive pieces.
  let pieces: Vec<&str> = prompt.split(IMAGE_TOKEN).collect();
  let placeholder_count = pieces.len() - 1;
  if placeholder_count != images.len() {
    return Err(crate::error::Error::ImageTokenCountMismatch {
      expected: images.len(), got: placeholder_count,
    });
  }

  let mut out = String::with_capacity(prompt.len() + 4096 * images.len());
  for (i, piece) in pieces.iter().enumerate() {
    out.push_str(piece);
    if i < images.len() {
      build_image_block(&mut out, &images[i]);
    }
  }
  Ok(out)
}

/// Build the per-image expanded block: `<|image_start|>` +
/// per-tile blocks + optional thumbnail + `<|image_end|>`.
fn build_image_block(out: &mut String, img: &crate::preproc::PreprocessedImage) {
  out.push_str(IMAGE_START);
  let rows = img.rows();
  let cols = img.cols();
  let tokens_per_main_tile = img.tokens_per_main_tile();

  if rows > 1 || cols > 1 {
    // Multi-tile path: per-tile position tokens.
    for r in 0..rows {
      for c in 0..cols {
        out.push('<');
        out.push('|');
        out.push_str("img_row_");
        out.push_str(&(r + 1).to_string());
        out.push_str("_col_");
        out.push_str(&(c + 1).to_string());
        out.push('|');
        out.push('>');
        for _ in 0..tokens_per_main_tile { out.push_str(IMAGE_TOKEN); }
      }
    }
    if let Some(thumb_tokens) = img.thumbnail_tokens() {
      out.push_str(IMAGE_THUMBNAIL);
      for _ in 0..thumb_tokens { out.push_str(IMAGE_TOKEN); }
    }
  } else {
    // Single-tile path: just repeat <image>.
    let total = img.num_image_tokens();
    for _ in 0..total { out.push_str(IMAGE_TOKEN); }
  }
  out.push_str(IMAGE_END);
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn special_token_ids_match_phase_0_capture() {
    assert_eq!(IMAGE_TOKEN_ID, 396);
    assert_eq!(EOS_TOKEN_ID, 7);
    assert_eq!(PAD_TOKEN_ID, 0);
    assert_eq!(BOS_TOKEN_ID, 1);
  }

  #[cfg(feature = "inference")]
  #[test]
  fn apply_chat_template_matches_upstream_fixtures() {
    let raw = include_str!("../tests/fixtures/chat_template_cases.json");
    let cases: serde_json::Value = serde_json::from_str(raw).unwrap();
    for case in cases.as_array().unwrap() {
      let name = case["name"].as_str().unwrap();
      // ... deserialize messages + tools + add_generation_prompt,
      //     call apply_chat_template, compare to case["expected"].
      // Test passes when all 8 cases match byte-for-byte.
    }
  }

  #[test]
  fn expand_image_placeholders_count_mismatch() {
    let r = expand_image_placeholders("Hello <image>", &[]);
    assert!(matches!(r, Err(crate::error::Error::ImageTokenCountMismatch { expected: 0, got: 1 })));
  }
  // More tests against image_expansion_cases.json — capture similarly.
}
```

- [ ] **Step 3: Run tests to verify they fail (then iterate Cargo.toml deps if needed)**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo test --lib chat_template::
```

Expected: FAIL — `apply_chat_template` not yet matching all upstream cases. Iterate the minijinja invocation until all 8 fixture cases pass byte-for-byte.

- [ ] **Step 4: Iterate until tests pass**

Common issues to fix:
- Jinja template uses `{%- generation -%}` / `{%- endgeneration -%}` — minijinja doesn't recognize these. Either preprocess them out of the source string before passing to `template_from_str`, or register a custom no-op tag handler. Easiest: `let preprocessed = BUNDLED_CHAT_TEMPLATE_JINJA.replace("{%- generation -%}", "").replace("{%- endgeneration -%}", "");` and use that.
- The template's `format_arg_value`/`render_tool_calls` macros use `tojson` — minijinja's `to_json` filter exists; configure with `env.add_filter("tojson", minijinja::filters::to_json)`.
- `namespace(...)` is supported by minijinja's stable feature set.

- [ ] **Step 5: Update lib.rs re-exports**

In `lfm/src/lib.rs`:

```rust
pub mod chat_template;
pub use chat_template::{
  BOS, BOS_TOKEN_ID, EOS_TOKEN_ID, IMAGE_END, IMAGE_START, IMAGE_TOKEN, IMAGE_TOKEN_ID,
  IMAGE_THUMBNAIL, IM_END, IM_START, PAD_TOKEN_ID, TOOL_CALL_END, TOOL_CALL_START,
  expand_image_placeholders,
};
#[cfg(feature = "inference")]
pub use chat_template::{apply_chat_template, ContentItem, Message, UserContent};
pub use chat_template::BUNDLED_CHAT_TEMPLATE_JINJA;
```

- [ ] **Step 6: Commit**

```bash
cd /Users/user/Develop/findit-studio/lfm
git add src/chat_template.rs src/lib.rs tests/fixtures/chat_template_cases.json
git commit -m "feat(chat_template): minijinja-backed apply_chat_template + image expansion

Per spec §6.4 + §8.4:
- Special token constants (BOS, EOS, IM_START/END, IMAGE_*, TOOL_CALL_*)
  with their IDs from the Phase 0 tokenizer.json capture
- BUNDLED_CHAT_TEMPLATE_JINJA: include_str! the bundled Jinja
- apply_chat_template (gated on inference): renders via minijinja with
  the {%- generation -%} tags pre-processed out (no-op for our purposes;
  spec §13.3 #12 caveat noted)
- expand_image_placeholders: walks <image> placeholders, substitutes
  per-image structure (<|image_start|> + per-tile <|img_row_R_col_C|> +
  <image> × tokens + optional <|img_thumbnail|> + <image> × thumb_tokens
  + <|image_end|>) for multi-tile, or just <image> × num_tokens for
  single-tile

Fixture: tests/fixtures/chat_template_cases.json with 8 cases captured
from upstream transformers Jinja rendering. All 8 match byte-for-byte.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 7: preproc/ — tile_grid + Preprocessor + EXIF helpers

**Files:**
- Create: `lfm/src/preproc/mod.rs`
- Create: `lfm/src/preproc/tile_grid.rs`
- Create: `lfm/tests/fixtures/tile_grid_cases.json` (capture from upstream)
- Create: `lfm/tests/fixtures/image_expansion_cases.json` (already partially in Task 6)
- Modify: `lfm/src/lib.rs` (add `pub mod preproc;` + re-exports)

- [ ] **Step 1: Capture tile-grid fixtures from upstream**

```bash
cd /Users/user/Develop/findit-studio/lfm/tests/fixtures
python3 - <<'PY' > tile_grid_cases.json
import json
from transformers import AutoImageProcessor
from PIL import Image
import numpy as np
proc = AutoImageProcessor.from_pretrained("LiquidAI/LFM2.5-VL-450M", trust_remote_code=True)
cases = []
for (w, h, label) in [
  (256, 256,   "small_square_single_tile"),
  (512, 512,   "exactly_one_tile_threshold"),
  (1024, 1024, "multi_tile_2x2"),
  (1920, 1080, "wide_16_9"),
  (1080, 1920, "tall_9_16"),
  (3840, 2160, "very_wide_4k"),
  (480, 640,   "portrait_3_4"),
  (640, 480,   "landscape_4_3"),
  (300, 300,   "below_threshold_single_tile"),
  (800, 600,   "near_threshold_4_3"),
  # ... add 10 more covering edge cases
]:
  img = Image.new("RGB", (w, h), (128, 128, 128))
  out = proc(images=[img], return_tensors="pt")
  case = {
    "name": label,
    "src_w": w, "src_h": h,
    "pixel_values_shape": list(out["pixel_values"].shape),
    "pixel_attention_mask_shape": list(out["pixel_attention_mask"].shape),
    "spatial_shapes": out["spatial_shapes"].cpu().numpy().tolist(),
  }
  cases.append(case)
print(json.dumps(cases, indent=2))
PY
```

- [ ] **Step 2: Write the failing test for tile_grid**

Create `lfm/src/preproc/tile_grid.rs`:

```rust
//! Tile-grid algorithm port of upstream `image_processing_lfm2_vl.py`.
//! Two paths per spec §8.3:
//! - **Multi-tile**: uniform 512×512 main tiles via `find_closest_aspect_ratio`
//!   + optional thumbnail dynamically sized via `smart_resize`.
//! - **Single-tile**: dynamically sized via `smart_resize`.

use crate::error::{Error, Result};
use crate::options::ImageBudget;

/// One image's tile-grid layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
  pub rows: u32,
  pub cols: u32,
  /// Main-tile size in pixels — (512, 512) in multi-tile path,
  /// dynamic in single-tile path.
  pub tile_h: u32,
  pub tile_w: u32,
  /// `Some((thumb_h, thumb_w))` only in multi-tile + use_thumbnail.
  pub thumbnail: Option<(u32, u32)>,
}

const PATCH_SIZE: u32 = 16;
const DOWNSAMPLE_FACTOR: u32 = 2;
const TILE_PIXEL_UNIT: u32 = PATCH_SIZE * DOWNSAMPLE_FACTOR;  // 32
const FULL_TILE_SIZE: u32 = 512;

/// Pick the tile grid for one image given its source dims and budget.
pub fn pick_tile_grid(src_w: u32, src_h: u32, budget: &ImageBudget) -> Result<TileGrid> {
  if src_w == 0 || src_h == 0 {
    return Err(Error::ImageTooSmall { w: src_w, h: src_h, min_w: 32, min_h: 32 });
  }
  budget.validate()?;
  let area = u64::from(src_w) * u64::from(src_h);
  let pixel_cap = u64::from(budget.max_image_tokens()) * u64::from(TILE_PIXEL_UNIT) * u64::from(TILE_PIXEL_UNIT);
  let tolerance = budget.max_pixels_tolerance();
  let multi_tile_threshold = (pixel_cap as f64 * tolerance as f64) as u64;

  if area > multi_tile_threshold {
    // ===== Multi-tile path =====
    let (rows, cols) = find_closest_aspect_ratio(
      src_w as f32 / src_h as f32, budget.min_tiles() as u32, budget.max_tiles() as u32, area,
    );
    let thumbnail = if budget.use_thumbnail() {
      let (tw, th) = smart_resize(src_w, src_h, budget.min_image_tokens(), budget.max_image_tokens());
      Some((th, tw))
    } else { None };
    Ok(TileGrid { rows, cols, tile_h: FULL_TILE_SIZE, tile_w: FULL_TILE_SIZE, thumbnail })
  } else {
    // ===== Single-tile path =====
    let (tw, th) = smart_resize(src_w, src_h, budget.min_image_tokens(), budget.max_image_tokens());
    Ok(TileGrid { rows: 1, cols: 1, tile_h: th, tile_w: tw, thumbnail: None })
  }
}

/// Enumerate `(rows, cols)` candidates with `rows*cols ∈ [min_tiles, max_tiles]`,
/// pick the one whose ratio is closest to `src_aspect`. Ties broken by
/// area match. Direct port of upstream `find_closest_aspect_ratio`.
fn find_closest_aspect_ratio(src_aspect: f32, min_tiles: u32, max_tiles: u32, src_area: u64) -> (u32, u32) {
  let mut best: Option<(u32, u32)> = None;
  let mut best_score = f32::INFINITY;
  for total in min_tiles..=max_tiles {
    for rows in 1..=total {
      if total % rows != 0 { continue; }
      let cols = total / rows;
      let aspect = cols as f32 / rows as f32;
      let diff = (aspect - src_aspect).abs();
      if diff < best_score {
        best_score = diff;
        best = Some((rows, cols));
      }
    }
  }
  best.unwrap_or((1, 1))
}

/// Resize (src_w, src_h) so:
/// 1. Both dims are multiples of `TILE_PIXEL_UNIT` (32).
/// 2. Total pixels ∈ [min_tokens × 32², max_tokens × 32²].
/// 3. Aspect ratio preserved via single scaling factor.
/// Returns `(width, height)` (NOT (h, w) — matches upstream order).
fn smart_resize(src_w: u32, src_h: u32, min_tokens: usize, max_tokens: usize) -> (u32, u32) {
  let unit = TILE_PIXEL_UNIT as u64;
  let unit_sq = unit * unit;
  let min_area = (min_tokens as u64) * unit_sq;
  let max_area = (max_tokens as u64) * unit_sq;
  let cur_area = u64::from(src_w) * u64::from(src_h);
  let beta: f64 = if cur_area > max_area {
    (max_area as f64 / cur_area as f64).sqrt()
  } else if cur_area < min_area {
    (min_area as f64 / cur_area as f64).sqrt()
  } else { 1.0 };
  let w = ((src_w as f64 * beta) / unit as f64).round() as u32 * unit as u32;
  let h = ((src_h as f64 * beta) / unit as f64).round() as u32 * unit as u32;
  (w.max(unit as u32), h.max(unit as u32))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn small_square_routes_to_single_tile() {
    let g = pick_tile_grid(256, 256, &ImageBudget::new()).unwrap();
    assert_eq!(g.rows, 1);
    assert_eq!(g.cols, 1);
    assert!(g.thumbnail.is_none());
  }

  #[test]
  fn large_square_routes_to_multi_tile() {
    let g = pick_tile_grid(1024, 1024, &ImageBudget::new()).unwrap();
    assert!(g.rows >= 1 && g.cols >= 1);
    assert!(g.rows * g.cols >= 2);
    assert_eq!(g.tile_h, 512);
    assert_eq!(g.tile_w, 512);
    assert!(g.thumbnail.is_some());
  }

  #[test]
  fn smart_resize_snaps_to_unit_multiples() {
    let (w, h) = smart_resize(1920, 1080, 64, 256);
    assert_eq!(w % TILE_PIXEL_UNIT, 0);
    assert_eq!(h % TILE_PIXEL_UNIT, 0);
  }

  #[test]
  fn upstream_fixture_parity() {
    let raw = include_str!("../../tests/fixtures/tile_grid_cases.json");
    let cases: serde_json::Value = serde_json::from_str(raw).unwrap();
    for case in cases.as_array().unwrap() {
      let src_w = case["src_w"].as_u64().unwrap() as u32;
      let src_h = case["src_h"].as_u64().unwrap() as u32;
      let g = pick_tile_grid(src_w, src_h, &ImageBudget::new()).unwrap();
      let expected_spatial: Vec<Vec<i64>> = serde_json::from_value(case["spatial_shapes"].clone()).unwrap();
      // Compare g.rows × g.cols against the number of spatial entries
      // (each entry is one tile; thumbnail adds one more if present).
      let main_tiles = (g.rows * g.cols) as usize;
      let expected_total = expected_spatial.len();
      let with_thumb_extra = if g.thumbnail.is_some() { 1 } else { 0 };
      assert_eq!(main_tiles + with_thumb_extra, expected_total,
        "case {}: main+thumb tile count mismatch (got {} + {}, expected {})",
        case["name"], main_tiles, with_thumb_extra, expected_total);
      // Spot-check spatial_shapes values for first main tile (h_patches, w_patches).
      let first = &expected_spatial[0];
      let h_patches = (g.tile_h / PATCH_SIZE) as i64;
      let w_patches = (g.tile_w / PATCH_SIZE) as i64;
      assert_eq!(first[0], h_patches, "case {}: tile h_patches", case["name"]);
      assert_eq!(first[1], w_patches, "case {}: tile w_patches", case["name"]);
    }
  }
}
```

- [ ] **Step 3: Run tile_grid tests**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo test --lib --no-default-features preproc::tile_grid::
```

Expected: PASS for the constructive tests; the upstream-parity test may need iteration on the smart_resize formula and find_closest_aspect_ratio scoring until it matches all 18+ fixtures.

- [ ] **Step 4: Write Preprocessor + PreprocessedImage**

Create `lfm/src/preproc/mod.rs`:

```rust
//! Image preprocessing for LFM2.5-VL. Wasm-compatible — no `ort` /
//! `tokenizers` deps. Pure Rust.

use std::path::Path;

use image::DynamicImage;

use crate::error::{Error, Result};
use crate::options::ImageBudget;

pub mod tile_grid;
pub use tile_grid::TileGrid;

/// Preprocessor for LFM2.5-VL inputs.
#[derive(Debug, Clone, Copy)]
pub struct Preprocessor {
  budget: ImageBudget,
}

impl Preprocessor {
  pub fn new(budget: ImageBudget) -> Self { Self { budget } }

  pub fn budget(&self) -> &ImageBudget { &self.budget }

  /// Single-image preprocess.
  pub fn preprocess(&self, image: &DynamicImage) -> Result<PreprocessedImage> {
    self.budget.validate()?;
    let (w, h) = (image.width(), image.height());
    let grid = tile_grid::pick_tile_grid(w, h, &self.budget)?;
    flatten_to_patches(image, &grid)
  }

  /// Multi-image convenience.
  pub fn preprocess_batch(&self, images: &[DynamicImage]) -> Result<Vec<PreprocessedImage>> {
    images.iter().map(|i| self.preprocess(i)).collect()
  }

  /// Path-based convenience with EXIF orientation correction.
  #[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
  pub fn preprocess_path(&self, path: &Path) -> Result<PreprocessedImage> {
    let img = decode_with_orientation(path)?;
    self.preprocess(&img)
  }
}

/// Output of `Preprocessor::preprocess` — directly fed to `vision_encoder.run`.
///
/// LAYOUT (Phase 0 G-confirmed):
/// - `pixel_values`: `[N_batch, num_patches, 768]` flattened (NOT image-shaped).
///   768 = 16² × 3 = patch_size² × channels.
/// - `pixel_attention_mask`: `[N_batch, num_patches]` — 1 = valid, 0 = padded.
/// - `spatial_shapes`: `[N_batch, 2]` — (h_patches, w_patches) per entry.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
  pixel_values: Vec<f32>,           // flattened [batch * num_patches * 768]
  pixel_attention_mask: Vec<i64>,   // [batch * num_patches]
  spatial_shapes: Vec<i64>,         // [batch * 2]
  batch_size: usize,                // N_batch
  patches_per_entry: usize,         // num_patches dim
  rows: u32,
  cols: u32,
  main_tile_h: u32,
  main_tile_w: u32,
  thumbnail_size: Option<(u32, u32)>,
  tokens_per_main_tile: usize,
  thumbnail_tokens: Option<usize>,
}

impl PreprocessedImage {
  pub fn pixel_values(&self) -> &[f32] { &self.pixel_values }
  pub fn pixel_attention_mask(&self) -> &[i64] { &self.pixel_attention_mask }
  pub fn spatial_shapes(&self) -> &[i64] { &self.spatial_shapes }
  pub fn batch_size(&self) -> usize { self.batch_size }
  pub fn patches_per_entry(&self) -> usize { self.patches_per_entry }
  pub fn num_tiles(&self) -> usize { (self.rows * self.cols) as usize + if self.thumbnail_size.is_some() { 1 } else { 0 } }
  pub fn rows(&self) -> usize { self.rows as usize }
  pub fn cols(&self) -> usize { self.cols as usize }
  pub fn main_tile_size(&self) -> (usize, usize) { (self.main_tile_h as usize, self.main_tile_w as usize) }
  pub fn thumbnail_size(&self) -> Option<(usize, usize)> { self.thumbnail_size.map(|(h, w)| (h as usize, w as usize)) }
  pub fn tokens_per_main_tile(&self) -> usize { self.tokens_per_main_tile }
  pub fn thumbnail_tokens(&self) -> Option<usize> { self.thumbnail_tokens }
  pub fn num_image_tokens(&self) -> usize {
    (self.rows as usize) * (self.cols as usize) * self.tokens_per_main_tile
      + self.thumbnail_tokens.unwrap_or(0)
  }
}

/// Decode an image from disk, applying EXIF orientation. Mirrors siglip2.
#[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
pub fn decode_with_orientation(path: &Path) -> Result<DynamicImage> {
  use image::{ImageDecoder, ImageReader};
  let mut decoder = ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
  let orientation = decoder.orientation()?;
  let mut img = DynamicImage::from_decoder(decoder)?;
  img.apply_orientation(orientation);
  Ok(img)
}

/// In-memory variant of `decode_with_orientation`.
#[cfg(feature = "decoders")]
pub fn decode_bytes_with_orientation(bytes: &[u8]) -> Result<DynamicImage> {
  use image::{ImageDecoder, ImageReader};
  use std::io::Cursor;
  let mut decoder = ImageReader::new(Cursor::new(bytes)).with_guessed_format()?.into_decoder()?;
  let orientation = decoder.orientation()?;
  let mut img = DynamicImage::from_decoder(decoder)?;
  img.apply_orientation(orientation);
  Ok(img)
}

/// Convert the source image into the patch-flattened tensor layout
/// the upstream Lfm2VlImageProcessor produces. Per the corrected §8.3:
/// each tile is 512×512 in multi-tile path; thumbnail dynamic.
fn flatten_to_patches(src: &DynamicImage, grid: &TileGrid) -> Result<PreprocessedImage> {
  use image::imageops;
  // 1. Resize source to (grid.cols * tile_w, grid.rows * tile_h).
  let target_w = grid.cols * grid.tile_w;
  let target_h = grid.rows * grid.tile_h;
  let resized = if src.width() == target_w && src.height() == target_h {
    src.clone().to_rgb8()
  } else {
    imageops::resize(&src.to_rgb8(), target_w, target_h, imageops::FilterType::Triangle)
  };

  // 2. Build per-tile RGB blocks (in row-major order).
  let mut tiles: Vec<image::RgbImage> = Vec::with_capacity(grid.num_main_tiles() + grid.thumbnail.is_some() as usize);
  for r in 0..grid.rows {
    for c in 0..grid.cols {
      let crop = imageops::crop_imm(&resized, c * grid.tile_w, r * grid.tile_h, grid.tile_w, grid.tile_h).to_image();
      tiles.push(crop);
    }
  }
  // 3. Append thumbnail tile (smart-resized version of the WHOLE source).
  if let Some((th, tw)) = grid.thumbnail {
    let thumb = imageops::resize(&src.to_rgb8(), tw, th, imageops::FilterType::Triangle);
    tiles.push(thumb);
  }

  // 4. Per tile: chunk into 16×16 RGB patches → flatten to 768-vec each → normalize px/255 → 2*px-1.
  // Pad each tile entry to per-image num_patches max with zeros + mark in attention mask.
  let max_patches = tiles.iter().map(|t| ((t.height() / PATCH_SIZE_U) * (t.width() / PATCH_SIZE_U)) as usize).max().unwrap_or(0);
  let n_batch = tiles.len();
  let mut pixel_values = vec![0f32; n_batch * max_patches * 768];
  let mut attn_mask = vec![0i64; n_batch * max_patches];
  let mut spatial = Vec::with_capacity(n_batch * 2);

  for (i, tile) in tiles.iter().enumerate() {
    let (tw, th) = (tile.width(), tile.height());
    let (h_patches, w_patches) = ((th / PATCH_SIZE_U), (tw / PATCH_SIZE_U));
    spatial.push(h_patches as i64);
    spatial.push(w_patches as i64);
    let mut patch_idx = 0usize;
    for py in 0..h_patches {
      for px in 0..w_patches {
        let mut k = 0usize;
        for dy in 0..PATCH_SIZE_U {
          for dx in 0..PATCH_SIZE_U {
            let pix = tile.get_pixel(px * PATCH_SIZE_U + dx, py * PATCH_SIZE_U + dy);
            for ch in 0..3 {
              let v = (pix[ch] as f32 / 255.0) * 2.0 - 1.0;  // mean=std=0.5 normalization
              pixel_values[i * max_patches * 768 + patch_idx * 768 + k] = v;
              k += 1;
            }
          }
        }
        attn_mask[i * max_patches + patch_idx] = 1;
        patch_idx += 1;
      }
    }
  }

  let tokens_per_main = ((grid.tile_h / TILE_PIXEL_UNIT_U) * (grid.tile_w / TILE_PIXEL_UNIT_U)) as usize;
  let thumbnail_tokens = grid.thumbnail.map(|(th, tw)| ((th / TILE_PIXEL_UNIT_U) * (tw / TILE_PIXEL_UNIT_U)) as usize);
  Ok(PreprocessedImage {
    pixel_values, pixel_attention_mask: attn_mask, spatial_shapes: spatial,
    batch_size: n_batch, patches_per_entry: max_patches,
    rows: grid.rows, cols: grid.cols,
    main_tile_h: grid.tile_h, main_tile_w: grid.tile_w,
    thumbnail_size: grid.thumbnail,
    tokens_per_main_tile: tokens_per_main, thumbnail_tokens,
  })
}

const PATCH_SIZE_U: u32 = 16;
const TILE_PIXEL_UNIT_U: u32 = 32;

impl TileGrid {
  fn num_main_tiles(&self) -> usize { (self.rows * self.cols) as usize }
}

#[cfg(test)]
mod tests {
  use super::*;
  use image::{ImageBuffer, Rgb};

  #[test]
  fn preprocess_small_square_succeeds() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(256, 256, Rgb([128, 128, 128])));
    let p = Preprocessor::new(ImageBudget::new());
    let out = p.preprocess(&img).unwrap();
    assert_eq!(out.batch_size(), 1);
    assert!(out.num_image_tokens() > 0);
  }

  #[test]
  fn preprocess_large_square_routes_multi_tile() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(1024, 1024, Rgb([128, 128, 128])));
    let p = Preprocessor::new(ImageBudget::new());
    let out = p.preprocess(&img).unwrap();
    assert!(out.num_tiles() >= 4);  // 2x2 main + maybe thumbnail
    assert_eq!(out.tokens_per_main_tile(), 256);
  }

  #[test]
  fn pixel_values_normalized_to_minus_one_to_one() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(256, 256, Rgb([255, 0, 0])));  // pure red
    let p = Preprocessor::new(ImageBudget::new());
    let out = p.preprocess(&img).unwrap();
    let pv = out.pixel_values();
    // First pixel of first patch: red channel should be 1.0 (255/255*2-1=1.0); g, b = -1.0.
    assert!((pv[0] - 1.0).abs() < 1e-5);
    assert!((pv[1] + 1.0).abs() < 1e-5);
    assert!((pv[2] + 1.0).abs() < 1e-5);
  }

  // More tests against tests/fixtures/tile_grid_cases.json's pixel_values_shape.
}
```

- [ ] **Step 5: Run preproc tests**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo test --lib preproc::
```

Expected: tests pass. Iterate until parity with upstream pixel_values_shape from fixtures.

- [ ] **Step 6: Update lib.rs**

```rust
pub mod preproc;
pub use preproc::{PreprocessedImage, Preprocessor, TileGrid};
#[cfg(feature = "decoders")]
pub use preproc::decode_bytes_with_orientation;
#[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
pub use preproc::decode_with_orientation;
```

- [ ] **Step 7: Commit**

```bash
cd /Users/user/Develop/findit-studio/lfm
git add src/preproc/ src/lib.rs tests/fixtures/tile_grid_cases.json
git commit -m "feat(preproc): tile_grid + Preprocessor + EXIF helpers

Per spec §6.4 + §8.3:
- preproc/tile_grid.rs: pick_tile_grid + find_closest_aspect_ratio +
  smart_resize. Multi-tile path uses uniform 512×512 main tiles + dynamic
  thumbnail; single-tile path uses smart_resize only.
- preproc/mod.rs: Preprocessor + PreprocessedImage with the Phase 0
  G-confirmed [N_batch, num_patches, 768] layout (pre-patchified, NOT
  image-shaped). Patch-level padding via attention_mask. Normalization
  px/255 → 2*px-1.
- decode_with_orientation (path) + decode_bytes_with_orientation (bytes)
  EXIF helpers — wasm-gated as appropriate.

Fixture parity with tests/fixtures/tile_grid_cases.json (18 cases
captured from upstream Lfm2VlImageProcessor).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 8: runtime/session.rs — build_session + check_outlet + validate_*_session

**Files:**
- Create: `lfm/src/runtime/mod.rs`
- Create: `lfm/src/runtime/session.rs`
- Modify: `lfm/src/lib.rs` (add `#[cfg(feature = "inference")] pub(crate) mod runtime;`)

- [ ] **Step 1: Skeleton runtime/mod.rs**

```rust
//! ORT-backed runtime modules. Gated on `feature = "inference"`.

pub(crate) mod session;
pub(crate) mod vision;
pub(crate) mod embed_tokens;
pub(crate) mod decoder;
pub(crate) mod sampler;
```

- [ ] **Step 2: Write the failing tests for check_outlet**

Create `lfm/src/runtime/session.rs`:

```rust
//! ORT session building + strict input/output validation.

use std::path::Path;

use ort::session::{Session, SessionBuilder};
use ort::value::{TensorElementType, ValueType};

use crate::error::{Error, Result};
use crate::options::Options;

/// Build an ORT session from a path with the given options.
pub(crate) fn build_session(graph: &Path, opts: &Options) -> Result<Session> {
  if !graph.exists() {
    return Err(Error::NotFound(graph.to_path_buf()));
  }
  let mut builder = Session::builder()?
    .with_optimization_level(opts.optimization_level())?;
  if let Some(t) = opts.thread().intra_threads() {
    builder = builder.with_intra_threads(t)?;
  }
  if let Some(t) = opts.thread().inter_threads() {
    builder = builder.with_inter_threads(t)?;
  }
  // EP registration is feature-gated — see siglip2 for the pattern.
  // (cuda/tensorrt/etc. blocks here, conditional on respective features.)
  let session = builder.commit_from_file(graph)?;
  Ok(session)
}

/// Verify a single outlet (input or output) matches the expected dtype + shape.
///
/// `expected_shape` semantics (mirrors siglip2):
/// - `-1` means "this axis MUST be dynamic in the graph".
/// - any other value means "exact match (or `-1` ok)".
pub(crate) fn check_outlet(
  outlets: &[ort::value::Outlet],
  name: &'static str,
  expected_dtype: TensorElementType,
  expected_shape: &[i64],
) -> Result<()> {
  let outlet = outlets.iter().find(|o| o.name() == name)
    .ok_or(Error::SessionShapeMismatch { input: name, expected: "outlet present in session", got: vec![] })?;
  match outlet.dtype() {
    ValueType::Tensor { ty, shape, .. } => {
      if *ty != expected_dtype {
        return Err(Error::SessionContractMismatch { input: name, expected: "matching tensor dtype", got: *ty });
      }
      let actual: &[i64] = shape;
      if actual.len() != expected_shape.len() {
        return Err(Error::SessionShapeMismatch { input: name, expected: "matching tensor rank", got: actual.to_vec() });
      }
      for (i, &want) in expected_shape.iter().enumerate() {
        let got = actual[i];
        if want == -1 {
          if got != -1 {
            return Err(Error::SessionShapeMismatch { input: name, expected: "dynamic axis required", got: actual.to_vec() });
          }
        } else if got != -1 && got != want {
          return Err(Error::SessionShapeMismatch { input: name, expected: "matching static dim", got: actual.to_vec() });
        }
      }
      Ok(())
    }
    _ => Err(Error::SessionShapeMismatch { input: name, expected: "tensor", got: vec![] }),
  }
}

/// Validate the vision encoder session against the Phase 0 contract.
pub(crate) fn validate_vision_session(s: &Session) -> Result<()> {
  check_outlet(s.inputs(),  "pixel_values",         TensorElementType::Float32, &[-1, -1, 768])?;
  check_outlet(s.inputs(),  "pixel_attention_mask", TensorElementType::Int64,   &[-1, -1])?;
  check_outlet(s.inputs(),  "spatial_shapes",       TensorElementType::Int64,   &[-1, 2])?;
  check_outlet(s.outputs(), "image_features",       TensorElementType::Float32, &[-1, 1024])?;
  Ok(())
}

pub(crate) fn validate_embed_session(s: &Session) -> Result<()> {
  check_outlet(s.inputs(),  "input_ids",     TensorElementType::Int64,   &[-1, -1])?;
  check_outlet(s.outputs(), "inputs_embeds", TensorElementType::Float32, &[-1, -1, 1024])?;
  Ok(())
}

/// Validate the decoder session against the Phase 0 contract.
/// Per G1 RESOLVED: NO position_ids input. Per G2/G3 RESOLVED: cache uses
/// SPARSE layer indices (conv at [0,1,3,4,6,7,9,11,13,15], attn at [2,5,8,10,12,14]).
pub(crate) fn validate_decoder_session(s: &Session) -> Result<()> {
  check_outlet(s.inputs(),  "inputs_embeds",  TensorElementType::Float32, &[-1, -1, 1024])?;
  check_outlet(s.inputs(),  "attention_mask", TensorElementType::Int64,   &[-1, -1])?;

  let cache = collect_cache_inputs(s.inputs())?;
  if cache.conv.len() != 10 || cache.attn.len() != 12 {
    return Err(Error::DecoderCacheMismatch {
      expected_conv: 10, expected_attn: 12,
      got_conv: cache.conv.len(), got_attn: cache.attn.len(),
    });
  }
  // Sparse-index check: collect indices from the discovered names and
  // verify they exactly match the expected sets.
  const EXPECTED_CONV: &[u32] = &[0, 1, 3, 4, 6, 7, 9, 11, 13, 15];
  const EXPECTED_ATTN: &[u32] = &[2, 5, 8, 10, 12, 14];
  let mut conv_indices: Vec<u32> = cache.conv.iter().filter_map(|n| parse_conv_index(n)).collect();
  conv_indices.sort_unstable();
  if conv_indices != EXPECTED_CONV {
    return Err(Error::SessionShapeMismatch {
      input: "past_conv.*", expected: "sparse indices [0,1,3,4,6,7,9,11,13,15]",
      got: conv_indices.into_iter().map(|i| i as i64).collect(),
    });
  }
  let mut attn_indices: Vec<u32> = cache.attn.iter().filter_map(|n| parse_attn_index(n)).collect();
  attn_indices.sort_unstable();
  attn_indices.dedup();
  if attn_indices != EXPECTED_ATTN {
    return Err(Error::SessionShapeMismatch {
      input: "past_key_values.*.{key,value}", expected: "sparse indices [2,5,8,10,12,14]",
      got: attn_indices.into_iter().map(|i| i as i64).collect(),
    });
  }
  check_outlet(s.outputs(), "logits", TensorElementType::Float32, &[-1, -1, 65536])?;
  Ok(())
}

pub(crate) struct CacheInputs {
  pub conv: Vec<String>,
  pub attn: Vec<String>,
}

pub(crate) fn collect_cache_inputs(outlets: &[ort::value::Outlet]) -> Result<CacheInputs> {
  let mut conv = Vec::new();
  let mut attn = Vec::new();
  for o in outlets {
    let n = o.name();
    if n.starts_with("past_conv.") { conv.push(n.to_string()); }
    else if n.starts_with("past_key_values.") { attn.push(n.to_string()); }
  }
  Ok(CacheInputs { conv, attn })
}

fn parse_conv_index(name: &str) -> Option<u32> {
  name.strip_prefix("past_conv.")?.parse().ok()
}
fn parse_attn_index(name: &str) -> Option<u32> {
  let rest = name.strip_prefix("past_key_values.")?;
  let dot = rest.find('.')?;
  rest[..dot].parse().ok()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parse_conv_index_works() {
    assert_eq!(parse_conv_index("past_conv.0"), Some(0));
    assert_eq!(parse_conv_index("past_conv.15"), Some(15));
    assert_eq!(parse_conv_index("past_kv.0"), None);
  }

  #[test]
  fn parse_attn_index_works() {
    assert_eq!(parse_attn_index("past_key_values.2.key"), Some(2));
    assert_eq!(parse_attn_index("past_key_values.14.value"), Some(14));
    assert_eq!(parse_attn_index("past_conv.0"), None);
  }

  // Tests that read from tests/fixtures/onnx_io_contract.json and verify
  // our validators would accept it. These don't require real ort sessions —
  // they reconstruct the ValueType from the JSON and call check_outlet directly.
  // (Implementer: write 6 tests, one per validate_* per session type.)
}
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo test --lib runtime::session::
```

Expected: parser tests pass. The fixture-based session-validation tests require either stubbing `ort::value::Outlet` (hard) or moving them to integration tests (easier — they need a real session anyway). Defer those to Task 11 (integration tests).

- [ ] **Step 4: Add lib.rs gate**

```rust
#[cfg(feature = "inference")]
pub(crate) mod runtime;
```

- [ ] **Step 5: Build to verify everything compiles**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo build --no-default-features
cargo build  # default features (inference + bundled + decoders)
```

- [ ] **Step 6: Commit**

```bash
cd /Users/user/Develop/findit-studio/lfm
git add src/runtime/ src/lib.rs
git commit -m "feat(runtime/session): build_session + check_outlet + validate_*_session

Per spec §8.5 + Phase 0 ONNX contract:
- build_session: ort SessionBuilder with optimization_level + threads
  from Options. EP registration deferred to per-EP feature blocks.
- check_outlet: same -1=required-dynamic semantics as siglip2.
- validate_vision_session: pixel_values [-1,-1,768] (pre-patchified),
  output 'image_features' [-1,1024] rank 2.
- validate_embed_session: input_ids → 'inputs_embeds' [-1,-1,1024].
- validate_decoder_session: NO position_ids check (G1). Sparse-index
  validation for past_conv (10 at [0,1,3,4,6,7,9,11,13,15]) and
  past_key_values (6 at [2,5,8,10,12,14]). Logits [-1,-1,65536].
- collect_cache_inputs + parse_{conv,attn}_index helpers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 9: runtime/vision.rs + runtime/embed_tokens.rs

**Files:**
- Create: `lfm/src/runtime/vision.rs`
- Create: `lfm/src/runtime/embed_tokens.rs`

These are thin ort-session wrappers. Both follow the same shape: hold `Session`, expose a `run` method that builds `TensorRef` inputs from `&PreprocessedImage`/`&[i64]` and extracts the typed output.

- [ ] **Step 1: Write VisionEncoder**

`lfm/src/runtime/vision.rs`:

```rust
//! VisionEncoder — wraps `vision_encoder.onnx`. Single-image only;
//! multi-image callers MUST loop (Phase 0 G6 RESOLVED — see spec §7.5).

use ort::session::Session;
use ort::value::TensorRef;

use crate::error::{Error, Result};
use crate::preproc::PreprocessedImage;
use crate::runtime::session::{build_session, validate_vision_session};
use crate::options::Options;

pub(crate) struct VisionEncoder { session: Session }

impl VisionEncoder {
  pub fn from_path(path: &std::path::Path, opts: &Options) -> Result<Self> {
    let session = build_session(path, opts)?;
    validate_vision_session(&session)?;
    Ok(Self { session })
  }

  pub fn from_session(session: Session) -> Result<Self> {
    validate_vision_session(&session)?;
    Ok(Self { session })
  }

  /// Run vision encoding on a single preprocessed image. Returns
  /// the raw image_features rows: `[num_image_tokens, 1024]` flattened
  /// + the row count.
  ///
  /// **Single-image only.** Per Phase 0 G6 + spec §7.5: batched calls
  /// across multiple images SILENTLY CORRUPT outputs when any image
  /// routes through the multi-tile path. Engine::run/generate iterate
  /// per-image and concatenate.
  pub fn run(&mut self, img: &PreprocessedImage) -> Result<Vec<f32>> {
    let pv_shape = [img.batch_size(), img.patches_per_entry(), 768];
    let mask_shape = [img.batch_size(), img.patches_per_entry()];
    let sp_shape = [img.batch_size(), 2];

    let pv = TensorRef::from_array_view((pv_shape, img.pixel_values()))?;
    let mask = TensorRef::from_array_view((mask_shape, img.pixel_attention_mask()))?;
    let sp = TensorRef::from_array_view((sp_shape, img.spatial_shapes()))?;

    let outputs = self.session.run(ort::inputs![
      "pixel_values" => pv,
      "pixel_attention_mask" => mask,
      "spatial_shapes" => sp,
    ])?;

    let out = outputs.get("image_features")
      .ok_or(Error::SessionShapeMismatch { input: "image_features", expected: "output present", got: vec![] })?;
    let (shape, data) = out.try_extract_tensor::<f32>()?;
    if shape.len() != 2 {
      return Err(Error::SessionShapeMismatch { input: "image_features", expected: "rank 2", got: shape.to_vec() });
    }
    if shape[1] != 1024 {
      return Err(Error::SessionShapeMismatch { input: "image_features", expected: "second dim 1024", got: shape.to_vec() });
    }
    Ok(data.to_vec())  // flat [num_image_tokens * 1024]
  }
}
```

- [ ] **Step 2: Write EmbedTokens**

`lfm/src/runtime/embed_tokens.rs`:

```rust
//! EmbedTokens — wraps `embed_tokens.onnx`.

use ort::session::Session;
use ort::value::TensorRef;

use crate::error::{Error, Result};
use crate::runtime::session::{build_session, validate_embed_session};
use crate::options::Options;

pub(crate) struct EmbedTokens { session: Session }

impl EmbedTokens {
  pub fn from_path(path: &std::path::Path, opts: &Options) -> Result<Self> {
    let session = build_session(path, opts)?;
    validate_embed_session(&session)?;
    Ok(Self { session })
  }

  pub fn from_session(session: Session) -> Result<Self> {
    validate_embed_session(&session)?;
    Ok(Self { session })
  }

  /// Embed a sequence of token IDs. Returns flattened `[seq_len * 1024]`.
  pub fn run(&mut self, input_ids: &[i64]) -> Result<Vec<f32>> {
    let shape = [1usize, input_ids.len()];
    let ids = TensorRef::from_array_view((shape, input_ids))?;
    let outputs = self.session.run(ort::inputs!["input_ids" => ids])?;
    let out = outputs.get("inputs_embeds")
      .ok_or(Error::SessionShapeMismatch { input: "inputs_embeds", expected: "output present", got: vec![] })?;
    let (s, data) = out.try_extract_tensor::<f32>()?;
    if s.len() != 3 || s[2] != 1024 {
      return Err(Error::SessionShapeMismatch { input: "inputs_embeds", expected: "[1, seq, 1024]", got: s.to_vec() });
    }
    Ok(data.to_vec())
  }
}
```

- [ ] **Step 3: Build to verify**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo build  # default features
```

- [ ] **Step 4: Commit**

```bash
cd /Users/user/Develop/findit-studio/lfm
git add src/runtime/vision.rs src/runtime/embed_tokens.rs
git commit -m "feat(runtime): VisionEncoder + EmbedTokens session wrappers

Both validate at construction via validate_*_session (Phase 0 contract).
VisionEncoder::run takes a single PreprocessedImage (per-image discipline
per G6); returns flat [num_image_tokens * 1024] image_features.
EmbedTokens::run takes &[i64] input_ids; returns flat [seq * 1024]
inputs_embeds. Both surface SessionShapeMismatch on output-shape drift.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 10: runtime/decoder.rs — Decoder + KvCache

**Files:**
- Create: `lfm/src/runtime/decoder.rs`

This is the densest runtime module. KvCache holds 22 tensors total (10 conv + 12 K/V) keyed by `SmolStr` (per spec §8.1).

- [ ] **Step 1: Write the failing tests for KvCache (no real session needed for the prefix-parse logic)**

```rust
#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn parse_present_to_past_conv() {
    let map = build_present_to_past(&["present_conv.0".into(), "present.2.key".into(), "present.2.value".into()]);
    assert_eq!(map.get("present_conv.0").map(SmolStr::as_str), Some("past_conv.0"));
    assert_eq!(map.get("present.2.key").map(SmolStr::as_str), Some("past_key_values.2.key"));
    assert_eq!(map.get("present.2.value").map(SmolStr::as_str), Some("past_key_values.2.value"));
  }
}
```

- [ ] **Step 2: Implement KvCache**

```rust
//! Decoder session wrapper + hybrid KV/conv cache management.

use std::collections::HashMap;

use ort::session::{Session, SessionOutputs};
use ort::value::{Tensor, TensorRef};
use smol_str::SmolStr;

use crate::error::{Error, Result};
use crate::runtime::session::{build_session, collect_cache_inputs, validate_decoder_session};
use crate::options::Options;

pub(crate) struct Decoder { session: Session, cache_init_template: KvCacheTemplate }

pub(crate) struct KvCacheTemplate {
  /// Names + shapes of `past_conv.{i}` inputs in the decoder.
  /// Shape is `[1, 1024, 3]` (all static — no `-1` here).
  conv: Vec<(SmolStr, Vec<i64>)>,
  /// Names + shapes of `past_key_values.{i}.{key,value}` inputs.
  /// Shape is `[1, 8, -1, 64]` where `-1` marks the dynamic past_len axis.
  attn: Vec<(SmolStr, Vec<i64>)>,
  /// Map present_X (output name) → past_X (input name).
  present_to_past: HashMap<SmolStr, SmolStr>,
}
// Both shape vecs use Vec<i64> for consistency. -1 means "dynamic axis"
// in either; new_cache resolves -1 → 0 for empty initialization.

pub(crate) struct KvCache {
  conv: HashMap<SmolStr, Tensor<f32>>,
  attn: HashMap<SmolStr, Tensor<f32>>,
  past_len: usize,
  template: KvCacheTemplate,
}

impl Decoder {
  pub fn from_path(path: &std::path::Path, opts: &Options) -> Result<Self> {
    let session = build_session(path, opts)?;
    validate_decoder_session(&session)?;
    let cache_init_template = build_cache_template(&session)?;
    Ok(Self { session, cache_init_template })
  }

  pub fn from_session(session: Session) -> Result<Self> {
    validate_decoder_session(&session)?;
    let cache_init_template = build_cache_template(&session)?;
    Ok(Self { session, cache_init_template })
  }

  pub fn new_cache(&self) -> Result<KvCache> {
    let mut conv = HashMap::new();
    for (name, shape_i64) in &self.cache_init_template.conv {
      // [1, 1024, 3] zero-fill — STATIC SHAPE per Phase 0.
      // No -1s in conv shapes; convert i64 → usize directly.
      let shape: Vec<usize> = shape_i64.iter().map(|&d| d as usize).collect();
      let tensor = Tensor::from_array((shape.as_slice(), vec![0f32; shape.iter().product()]))?;
      conv.insert(name.clone(), tensor);
    }
    let mut attn = HashMap::new();
    for (name, shape_i64) in &self.cache_init_template.attn {
      // [1, 8, -1, 64] template with -1 → 0 for empty initialization.
      let shape: Vec<usize> = shape_i64.iter().map(|&d| if d < 0 { 0 } else { d as usize }).collect();
      let tensor = Tensor::from_array((shape.as_slice(), vec![0f32; shape.iter().product()]))?;
      attn.insert(name.clone(), tensor);
    }
    Ok(KvCache { conv, attn, past_len: 0, template: self.cache_init_template.clone() })
  }

  /// One step of decoding. Mutates cache in place.
  /// Returns flattened logits — last position of seq for prefill,
  /// or the single position for incremental decode.
  pub fn step(
    &mut self,
    cache: &mut KvCache,
    inputs_embeds: &[f32],
    seq_len: usize,
  ) -> Result<Vec<f32>> {
    let total_len = cache.past_len + seq_len;
    let attn_mask = vec![1i64; total_len];
    let inputs_shape = [1usize, seq_len, 1024];

    let embeds_ref = TensorRef::from_array_view((inputs_shape, inputs_embeds))?;
    let mask_ref = TensorRef::from_array_view(([1usize, total_len], attn_mask.as_slice()))?;

    let mut session_inputs = vec![
      ("inputs_embeds", embeds_ref),
      ("attention_mask", mask_ref),
    ];
    // Append cache inputs (pass-by-borrow; lifetime tied to `cache`).
    let mut cache_refs: Vec<(String, TensorRef<'_, f32>)> = Vec::new();
    for (name, tensor) in cache.conv.iter().chain(cache.attn.iter()) {
      cache_refs.push((name.to_string(), tensor.view()));
    }
    for (name, view) in &cache_refs {
      session_inputs.push((name.as_str(), view.clone()));
    }

    let outputs = self.session.run(session_inputs)?;
    let logits_out = outputs.get("logits")
      .ok_or(Error::SessionShapeMismatch { input: "logits", expected: "output present", got: vec![] })?;
    let (shape, data) = logits_out.try_extract_tensor::<f32>()?;
    let last_pos = shape[1] as usize - 1;
    let vocab = shape[2] as usize;
    let logits = data[last_pos * vocab .. (last_pos + 1) * vocab].to_vec();

    // Advance the cache: present_X → past_X.
    advance_cache(cache, &outputs, &self.cache_init_template.present_to_past)?;
    cache.past_len = total_len;
    Ok(logits)
  }
}

fn build_cache_template(session: &Session) -> Result<KvCacheTemplate> {
  let cache = collect_cache_inputs(session.inputs())?;
  let mut conv = Vec::new();
  for name in cache.conv {
    // Static shape from input metadata: [batch=1, hidden=1024, conv_L_cache=3].
    // Stored as Vec<i64> for consistency with attn shapes (-1 sentinel reserved).
    conv.push((SmolStr::from(name), vec![1i64, 1024, 3]));
  }
  let mut attn = Vec::new();
  for name in cache.attn {
    // Dynamic past_len — start at 0.
    attn.push((SmolStr::from(name), vec![1, 8, -1, 64]));
  }
  let present_to_past = build_present_to_past(
    &session.outputs().iter().map(|o| SmolStr::from(o.name())).collect::<Vec<_>>(),
  );
  Ok(KvCacheTemplate { conv, attn, present_to_past })
}

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

fn advance_cache(cache: &mut KvCache, outputs: &SessionOutputs, present_to_past: &HashMap<SmolStr, SmolStr>) -> Result<()> {
  for (present_name, past_name) in present_to_past {
    if let Some(out) = outputs.get(present_name.as_str()) {
      let (shape, data) = out.try_extract_tensor::<f32>()?;
      let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
      let new_tensor = Tensor::from_array((shape.as_slice(), data.to_vec()))?;
      // Route to conv or attn map by past-name prefix.
      if past_name.starts_with("past_conv.") {
        cache.conv.insert(past_name.clone(), new_tensor);
      } else {
        cache.attn.insert(past_name.clone(), new_tensor);
      }
    }
  }
  Ok(())
}

impl Clone for KvCacheTemplate {
  fn clone(&self) -> Self {
    Self { conv: self.conv.clone(), attn: self.attn.clone(), present_to_past: self.present_to_past.clone() }
  }
}
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo test --lib runtime::decoder::
cargo build
```

Expected: PASS for the prefix-parse test. Build clean.

- [ ] **Step 4: Commit**

```bash
cd /Users/user/Develop/findit-studio/lfm
git add src/runtime/decoder.rs
git commit -m "feat(runtime/decoder): Decoder + KvCache with sparse layer indices

Per spec §8.1 + Phase 0 G2/G3:
- KvCache uses HashMap<SmolStr, Tensor<f32>> (NOT &'static str — names
  are runtime-discovered from session.inputs()).
- conv cache: 10 entries at sparse indices [0,1,3,4,6,7,9,11,13,15];
  STATIC shape [1, 1024, 3] zero-init at step 0.
- attn cache: 12 entries (6 layers × {key,value}) at sparse [2,5,8,10,12,14];
  DYNAMIC shape [1, 8, past_len, 64] with past_len=0 at step 0.
- present_to_past name map built from session.outputs() prefix dispatch:
  present_conv.* → past_conv.*, present.*.{key,value} → past_key_values.*.{key,value}.
- Decoder::step runs one decoder pass + advances cache in place.
  No position_ids input (G1 RESOLVED).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 11: runtime/sampler.rs — Sampler trait + FreeSampler + ConstrainedSampler

**Files:**
- Create: `lfm/src/runtime/sampler.rs`

- [ ] **Step 1: Implement the trait + both samplers**

```rust
//! Sampling for the decode loop. FreeSampler for `generate`,
//! ConstrainedSampler for `run` (uses llguidance for schema masking).

use llguidance::{Constraint, ParserFactory};
use smol_str::SmolStr;

use crate::error::{Error, Result};
use crate::options::RequestOptions;

pub(crate) trait Sampler {
  fn sample(&mut self, logits: &mut [f32]) -> Result<u32>;
  fn is_complete(&self) -> bool;
}

pub(crate) struct FreeSampler {
  request: RequestOptions,
  generated: Vec<u32>,
}

impl FreeSampler {
  pub fn new(request: RequestOptions) -> Self {
    Self { request, generated: Vec::with_capacity(512) }
  }
}

impl Sampler for FreeSampler {
  fn sample(&mut self, logits: &mut [f32]) -> Result<u32> {
    apply_repetition_penalty(logits, &self.generated, self.request.repetition_penalty());
    let token = if self.request.temperature() == 0.0 {
      argmax(logits)
    } else {
      sample_min_p(logits, self.request.temperature(), self.request.min_p())
    };
    self.generated.push(token);
    Ok(token)
  }
  fn is_complete(&self) -> bool { false }  // EOS / max_new_tokens decide for free-form
}

pub(crate) struct ConstrainedSampler {
  constraint: Constraint,
  request: RequestOptions,
  generated: Vec<u32>,
}

impl ConstrainedSampler {
  pub fn new(factory: &ParserFactory, schema: &serde_json::Value, request: RequestOptions) -> Result<Self> {
    let parser = factory.create_parser_from_lark_or_grammar(/* schema-as-grammar */)
      .map_err(Error::llguidance)?;
    let constraint = Constraint::new(parser);
    Ok(Self { constraint, request, generated: Vec::with_capacity(512) })
  }
}

impl Sampler for ConstrainedSampler {
  fn sample(&mut self, logits: &mut [f32]) -> Result<u32> {
    let mask = self.constraint.compute_mask().map_err(Error::llguidance)?;
    if mask_is_all_zero(&mask) {
      return Err(Error::LlGuidanceDeadEnd {
        step: self.generated.len(),
        state: SmolStr::from(format!("{:?}", self.constraint)),  // best-effort debug
      });
    }
    apply_mask_in_place(logits, &mask);
    apply_repetition_penalty(logits, &self.generated, self.request.repetition_penalty());
    let token = if self.request.temperature() == 0.0 {
      argmax(logits)
    } else {
      sample_min_p(logits, self.request.temperature(), self.request.min_p())
    };
    self.constraint.commit_token(token).map_err(Error::llguidance)?;
    self.generated.push(token);
    Ok(token)
  }
  fn is_complete(&self) -> bool { self.constraint.is_stopped() }
}

// ===== Sampling primitives =====

fn argmax(logits: &[f32]) -> u32 {
  let (idx, _) = logits.iter().enumerate()
    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
  idx as u32
}

fn sample_min_p(logits: &mut [f32], temperature: f32, min_p: f32) -> u32 {
  // 1. Apply temperature.
  for x in logits.iter_mut() { *x /= temperature; }
  // 2. Softmax.
  let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  let mut sum = 0.0f32;
  for x in logits.iter_mut() { *x = (*x - max).exp(); sum += *x; }
  for x in logits.iter_mut() { *x /= sum; }
  // 3. min_p threshold.
  let pmax = logits.iter().copied().fold(0.0f32, f32::max);
  let threshold = min_p * pmax;
  for x in logits.iter_mut() { if *x < threshold { *x = 0.0; } }
  // 4. Renormalize + categorical sample.
  let s: f32 = logits.iter().sum();
  let mut r = rand::random::<f32>() * s;
  for (i, &p) in logits.iter().enumerate() {
    r -= p;
    if r <= 0.0 { return i as u32; }
  }
  argmax(logits)
}

fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
  if penalty == 1.0 { return; }
  for &t in generated {
    if let Some(x) = logits.get_mut(t as usize) {
      if *x > 0.0 { *x /= penalty; } else { *x *= penalty; }
    }
  }
}

fn apply_mask_in_place(logits: &mut [f32], mask: &[u8]) {
  for (i, &m) in mask.iter().enumerate() {
    if m == 0 { logits[i] = f32::NEG_INFINITY; }
  }
}

fn mask_is_all_zero(mask: &[u8]) -> bool {
  mask.iter().all(|&m| m == 0)
}
```

**Cargo.toml edit (mandatory before this task compiles):** add `rand` as an optional dep + extend the `inference` feature to pull it. Use `rand = "0.9"` (current major as of 2026; siglip2/egemma don't pin rand, so we're picking ourselves — verify against latest at impl time). Edit `lfm/Cargo.toml`:

```toml
[dependencies]
# ... existing deps ...
rand        = { version = "0.9", optional = true }   # for sample_min_p (FreeSampler/ConstrainedSampler)

[features]
inference   = ["dep:ort", "dep:tokenizers", "dep:llguidance", "dep:minijinja", "dep:rand"]
```

**llguidance API caveat (P1-3):** the exact `Constraint::compute_mask` return type — `&[u8]` in this plan's pseudocode — is unverified against llguidance 1.7. The actual type is likely a packed bitmask (`SimpleVob` / `Vec<u32>`) with a `is_allowed(token_id) -> bool` accessor. Read llguidance 1.7 docs and replace `apply_mask_in_place` with the canonical bit-test loop. Same caveat applies to `ParserFactory::create_parser_*` — use whatever the docs name as the JSON-schema entry point.

- [ ] **Step 2: Add rand dep + run tests**

```bash
cd /Users/user/Develop/findit-studio/lfm
# Add `rand = { version = "0.8", optional = true }` and append to inference feature.
cargo build
cargo test --lib runtime::sampler::
```

- [ ] **Step 3: Commit**

```bash
git add src/runtime/sampler.rs Cargo.toml Cargo.lock
git commit -m "feat(runtime/sampler): Sampler trait + FreeSampler + ConstrainedSampler

Per spec §8.2 + §8.6:
- Sampler trait: sample(&mut [f32]) -> Result<u32> + is_complete().
- FreeSampler: rep-penalty + (greedy or min_p sampling).
- ConstrainedSampler: llguidance Constraint with compute_mask +
  commit_token. Empty-mask → Error::LlGuidanceDeadEnd. is_stopped() for
  schema-complete short-circuit.
- min_p sampling: temperature → softmax → min_p threshold → renormalize
  → categorical. Matches LFM2.5-VL model card recommendation (min_p,
  NOT top_p/top_k).
- Repetition penalty applied per upstream convention (divide on positive
  logits, multiply on negative).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 12: generate.rs — end-to-end pipeline

**Files:**
- Create: `lfm/src/generate.rs`

This is the core orchestration. Spec §7 describes the flow; key points: per-image vision encoder calls (G6), embed-merge writes flat rows (rank-2 image_features), no position_ids in decoder.

- [ ] **Step 1: Skeleton with the public-facing function signatures**

```rust
//! End-to-end generation pipeline. Glues preprocessing, chat-template
//! rendering, encoding, embed-merge, and decode loop together.

use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::chat_template::{self, IMAGE_TOKEN_ID, EOS_TOKEN_ID, IMAGE_TOKEN};
use crate::error::{Error, Result};
use crate::options::RequestOptions;
use crate::preproc::{PreprocessedImage, Preprocessor};
use crate::runtime::decoder::{Decoder, KvCache};
use crate::runtime::embed_tokens::EmbedTokens;
use crate::runtime::sampler::{ConstrainedSampler, FreeSampler, Sampler};
use crate::runtime::vision::VisionEncoder;
use llguidance::ParserFactory;
use vlm_tasks::Task;

/// Configuration for one generate/run call.
pub(crate) struct GenerateContext<'a> {
  pub preprocessor: &'a Preprocessor,
  pub tokenizer: &'a Tokenizer,
  pub vision: &'a mut VisionEncoder,
  pub embed: &'a mut EmbedTokens,
  pub decoder: &'a mut Decoder,
  pub request: &'a RequestOptions,
}

/// Free-form generation. Returns the decoded text.
pub(crate) fn generate(
  ctx: &mut GenerateContext<'_>,
  images: &[DynamicImage],
  prompt: &str,
) -> Result<String> {
  let mut sampler = FreeSampler::new(*ctx.request);
  run_pipeline(ctx, images, prompt, &mut sampler)
}

/// Structured-output run (uses llguidance via ConstrainedSampler).
pub(crate) fn run<T: Task>(
  ctx: &mut GenerateContext<'_>,
  factory: &ParserFactory,
  task: &T,
  images: &[DynamicImage],
) -> Result<T::Output> {
  let mut sampler = ConstrainedSampler::new(factory, task.schema(), *ctx.request)?;
  let text = run_pipeline(ctx, images, task.prompt(), &mut sampler)?;
  Ok(task.parse(&text)?)
}

fn run_pipeline<S: Sampler>(
  ctx: &mut GenerateContext<'_>,
  images: &[DynamicImage],
  prompt: &str,
  sampler: &mut S,
) -> Result<String> {
  ctx.request.validate()?;

  // 1. Preprocess each image (per-image, not batched — Phase 0 G6 RESOLVED).
  let preprocessed: Vec<PreprocessedImage> = ctx.preprocessor.preprocess_batch(images)?;

  // 2. Build chat-formatted prompt with literal <image> placeholders, then expand.
  let messages = build_messages(images.len(), prompt);
  let chat_text = chat_template::apply_chat_template(&messages, None, true)?;
  let expanded = chat_template::expand_image_placeholders(&chat_text, &preprocessed)?;

  // 3. Tokenize.
  let encoding = ctx.tokenizer.encode(expanded, false).map_err(Error::tokenizer)?;
  let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&u| u as i64).collect();

  // 4a. Embed text tokens.
  let mut inputs_embeds = ctx.embed.run(&input_ids)?;  // [seq * 1024]
  let seq_len = input_ids.len();

  // 4b. Encode each image individually (G6 contract — see spec §7.5).
  let mut image_features_total: Vec<f32> = Vec::new();
  for img in &preprocessed {
    let features = ctx.vision.run(img)?;  // [num_image_tokens_i * 1024]
    image_features_total.extend(features);
  }

  // 5. Embed-merge: walk input_ids, replace IMAGE_TOKEN_ID positions with image_features rows.
  let mut k = 0usize;
  for (pos, &tok) in input_ids.iter().enumerate() {
    if tok == IMAGE_TOKEN_ID as i64 {
      let dst = pos * 1024 .. (pos + 1) * 1024;
      let src = k * 1024 .. (k + 1) * 1024;
      inputs_embeds[dst].copy_from_slice(&image_features_total[src]);
      k += 1;
    }
  }
  if k * 1024 != image_features_total.len() {
    return Err(Error::ImageTokenCountMismatch {
      expected: image_features_total.len() / 1024, got: k,
    });
  }

  // 6. Decode loop.
  let mut cache = ctx.decoder.new_cache()?;
  let mut generated: Vec<u32> = Vec::with_capacity(ctx.request.max_new_tokens());

  // Step 0: prefill.
  let mut logits = ctx.decoder.step(&mut cache, &inputs_embeds, seq_len)?;
  if sampler.is_complete() {
    return Ok(decode_tokens(ctx.tokenizer, &generated)?);
  }
  let mut next = sampler.sample(&mut logits)?;
  if next == EOS_TOKEN_ID { return Ok(decode_tokens(ctx.tokenizer, &generated)?); }
  generated.push(next);

  // Steps k > 0.
  for _step in 1..ctx.request.max_new_tokens() {
    if sampler.is_complete() { break; }
    let next_embed = ctx.embed.run(&[next as i64])?;
    let mut logits_k = ctx.decoder.step(&mut cache, &next_embed, 1)?;
    next = sampler.sample(&mut logits_k)?;
    if next == EOS_TOKEN_ID { break; }
    generated.push(next);
  }

  if generated.is_empty() {
    return Err(Error::Empty);
  }
  if generated.len() == ctx.request.max_new_tokens() && !sampler.is_complete() {
    return Err(Error::MaxTokensExceeded {
      max: ctx.request.max_new_tokens(),
      schema_complete: sampler.is_complete(),
    });
  }

  decode_tokens(ctx.tokenizer, &generated)
}

fn build_messages(num_images: usize, prompt: &str) -> Vec<chat_template::Message<'_>> {
  use chat_template::{ContentItem, Message, UserContent};
  if num_images == 0 {
    vec![Message::User { content: UserContent::Text(prompt) }]
  } else {
    let mut items: Vec<ContentItem<'_>> = Vec::with_capacity(num_images + 1);
    for _ in 0..num_images { items.push(ContentItem::Image); }
    items.push(ContentItem::Text { text: prompt });
    vec![Message::User { content: UserContent::Multimodal(items) }]
  }
}

fn decode_tokens(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
  tokenizer.decode(ids, true).map_err(Error::tokenizer)
}
```

- [ ] **Step 2: Build to verify**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo build
```

Iterate on lifetime errors / borrow conflicts in the GenerateContext struct.

- [ ] **Step 3: Commit**

```bash
git add src/generate.rs
git commit -m "feat(generate): end-to-end pipeline (per-image vision calls)

Per spec §7:
1. preprocess_batch (per-image)
2. apply_chat_template + expand_image_placeholders
3. tokenize
4a. embed_tokens.run (single batched call for all text tokens)
4b. PER-IMAGE vision_encoder.run (G6 contract — never batched
    across images; concatenate image_features in source order)
5. embed-merge: walk input_ids, replace IMAGE_TOKEN_ID positions
   with image_features rows (rank-2 → flat slice copies)
6. decode loop with prefill (step 0) + incremental (k > 0):
   - Stop on EOS / max_new_tokens / sampler.is_complete()
   - No position_ids passed to decoder (G1)
7. detokenize (skip_special_tokens=true)

Pipeline parameterised over Sampler trait so generate (FreeSampler)
and run (ConstrainedSampler) share the same loop body.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 13: engine.rs — public Engine

**Files:**
- Create: `lfm/src/engine.rs`
- Modify: `lfm/src/lib.rs` (re-exports)

- [ ] **Step 1: Implement Engine**

```rust
//! Public `Engine` — the crate's main entry point.

use std::path::Path;

use image::DynamicImage;
use llguidance::ParserFactory;
use ort::session::Session;
use tokenizers::Tokenizer;
use tracing::{debug, info, instrument};
use vlm_tasks::Task;

use crate::error::{Error, Result};
use crate::generate::{self, GenerateContext};
use crate::options::{Options, RequestOptions, ImageBudget, ThreadOptions};
use crate::preproc::Preprocessor;
use crate::runtime::decoder::Decoder;
use crate::runtime::embed_tokens::EmbedTokens;
use crate::runtime::vision::VisionEncoder;

/// LFM2.5-VL inference engine.
///
/// `Engine: Send + !Sync` — `ort::Session` is `!Sync`. Workers wanting
/// parallelism instantiate one Engine per thread, or share one behind
/// `Mutex<Engine>`.
///
/// **Model weights license:** the model this Engine wraps ships under
/// the LFM Open License v1.0 (<https://www.liquid.ai/lfm-license>),
/// separate from this crate's MIT/Apache-2.0 license. Verify your use
/// case complies with Liquid AI's terms.
pub struct Engine {
  vision: VisionEncoder,
  embed: EmbedTokens,
  decoder: Decoder,
  tokenizer: Tokenizer,
  preprocessor: Preprocessor,
  options: Options,
  parser_factory: Option<ParserFactory>,
}

impl Engine {
  /// Construct from three ONNX file paths + a tokenizer.json path.
  /// Wasm-incompatible (uses ort's `commit_from_file`).
  #[cfg(not(target_arch = "wasm32"))]
  #[instrument(name = "lfm::Engine::from_files", skip(opts))]
  pub fn from_files(
    vision_onnx: &Path, embed_onnx: &Path, decoder_onnx: &Path,
    tokenizer_json: &Path, opts: Options,
  ) -> Result<Self> {
    let vision = VisionEncoder::from_path(vision_onnx, &opts)?;
    let embed = EmbedTokens::from_path(embed_onnx, &opts)?;
    let decoder = Decoder::from_path(decoder_onnx, &opts)?;
    let tokenizer = Tokenizer::from_file(tokenizer_json).map_err(Error::tokenizer)?;
    Ok(Self {
      vision, embed, decoder, tokenizer,
      preprocessor: Preprocessor::new(*opts.image_budget()),
      options: opts,
      parser_factory: None,
    })
  }

  /// Same as `from_files` but uses the bundled `tokenizer.json`.
  #[cfg(all(feature = "bundled", not(target_arch = "wasm32")))]
  #[instrument(name = "lfm::Engine::bundled", skip(opts))]
  pub fn bundled(vision_onnx: &Path, embed_onnx: &Path, decoder_onnx: &Path, opts: Options) -> Result<Self> {
    let vision = VisionEncoder::from_path(vision_onnx, &opts)?;
    let embed = EmbedTokens::from_path(embed_onnx, &opts)?;
    let decoder = Decoder::from_path(decoder_onnx, &opts)?;
    let tokenizer = Tokenizer::from_bytes(crate::BUNDLED_TOKENIZER).map_err(Error::tokenizer)?;
    Ok(Self {
      vision, embed, decoder, tokenizer,
      preprocessor: Preprocessor::new(*opts.image_budget()),
      options: opts,
      parser_factory: None,
    })
  }

  /// From caller-built sessions + tokenizer with crate-default Options.
  pub fn from_ort_sessions(
    vision: Session, embed: Session, decoder: Session, tokenizer: Tokenizer,
  ) -> Result<Self> {
    Self::from_ort_sessions_with_options(vision, embed, decoder, tokenizer, Options::new())
  }

  /// From caller-built sessions with custom Options. Mirrors siglip2/egemma.
  pub fn from_ort_sessions_with_options(
    vision: Session, embed: Session, decoder: Session, tokenizer: Tokenizer, opts: Options,
  ) -> Result<Self> {
    Ok(Self {
      vision: VisionEncoder::from_session(vision)?,
      embed: EmbedTokens::from_session(embed)?,
      decoder: Decoder::from_session(decoder)?,
      tokenizer,
      preprocessor: Preprocessor::new(*opts.image_budget()),
      options: opts,
      parser_factory: None,
    })
  }

  pub fn options(&self) -> &Options { &self.options }
  pub fn request(&self) -> &RequestOptions { self.options.request() }
  pub fn image_budget(&self) -> &ImageBudget { self.options.image_budget() }

  /// Warm up: 4-token generation against a 1024×1024 dummy black image.
  /// Cost: 2-5 seconds on CPU. Call once at startup.
  #[instrument(name = "lfm::Engine::warmup", skip(self))]
  pub fn warmup(&mut self) -> Result<()> {
    let started = std::time::Instant::now();
    let dummy = DynamicImage::ImageRgb8(image::ImageBuffer::new(1024, 1024));
    let opts = RequestOptions::new().with_max_new_tokens(4);
    let _ = self.generate_with(&[dummy], "ok", &opts);
    debug!(elapsed_ms = started.elapsed().as_millis() as u64, "warmup complete");
    Ok(())
  }

  /// Free-form text generation. Latency: 5-30 seconds on CPU.
  #[instrument(name = "lfm::Engine::generate", skip(self, images, prompt))]
  pub fn generate(&mut self, images: &[DynamicImage], prompt: &str) -> Result<String> {
    self.generate_with(images, prompt, &self.options.request().clone())
  }

  pub fn generate_with(&mut self, images: &[DynamicImage], prompt: &str, request: &RequestOptions) -> Result<String> {
    let mut ctx = GenerateContext {
      preprocessor: &self.preprocessor,
      tokenizer: &self.tokenizer,
      vision: &mut self.vision,
      embed: &mut self.embed,
      decoder: &mut self.decoder,
      request,
    };
    generate::generate(&mut ctx, images, prompt)
  }

  /// Structured output via `Task` trait.
  #[instrument(name = "lfm::Engine::run", skip(self, task, images), fields(task_kind = std::any::type_name::<T>()))]
  pub fn run<T: Task>(&mut self, task: &T, images: &[DynamicImage]) -> Result<T::Output> {
    self.run_with(task, images, &self.options.request().clone())
  }

  pub fn run_with<T: Task>(&mut self, task: &T, images: &[DynamicImage], request: &RequestOptions) -> Result<T::Output> {
    if self.parser_factory.is_none() {
      info!("building llguidance ParserFactory (one-time)");
      self.parser_factory = Some(build_parser_factory(&self.tokenizer)?);
    }
    let factory = self.parser_factory.as_ref().unwrap();
    let mut ctx = GenerateContext {
      preprocessor: &self.preprocessor,
      tokenizer: &self.tokenizer,
      vision: &mut self.vision,
      embed: &mut self.embed,
      decoder: &mut self.decoder,
      request,
    };
    generate::run(&mut ctx, factory, task, images)
  }
}

fn build_parser_factory(tokenizer: &Tokenizer) -> Result<ParserFactory> {
  // STEP-0 FOR THE IMPLEMENTER: this placeholder body MUST be replaced
  // with the canonical llguidance 1.7 ParserFactory construction before
  // any Engine::run integration test (Task 15) can run. All four of:
  //   structured_scene_task
  //   deterministic_run_is_idempotent
  //   max_tokens_cap_returns_max_tokens_exceeded
  //   schema_stops_at_closing_brace_not_max_tokens
  //   over_constrained_schema_returns_dead_end
  // hit this code path on first call and panic with `not yet implemented`
  // until it's wired up. The plan flags this as the highest-priority
  // unknown — fix it FIRST in Task 13 before moving to Task 14.
  //
  // What you need from llguidance 1.7 docs (https://docs.rs/llguidance/1.7):
  //   - The TokEnv abstraction (likely llguidance::tok_env::TokEnv,
  //     constructed from a tokenizers::Tokenizer via a builder).
  //   - ParserFactory::new(tok_env) or ::with_tokenizer(...) — confirm.
  //   - The schema → grammar compilation entry point (likely
  //     ParserFactory::create_parser(schema_value) or .json_schema(...)).
  //   - The Constraint::compute_mask return type — almost certainly NOT
  //     &[u8]; likely SimpleVob or Vec<u32> packed bitmask. Update
  //     runtime/sampler.rs::apply_mask_in_place to match the real type.
  //
  // Implementer: read the docs, write 5 lines, delete this comment.
  todo!("build llguidance ParserFactory from tokenizer per llguidance 1.7 docs — see comment above")
}
```

> **PLAN-EXECUTION NOTE (P1-3 + P2-3):** the llguidance integration in Tasks 11
> and 13 has TWO unverified-against-docs surfaces. Resolve them BEFORE moving
> past Task 13:
> 1. **`Constraint::compute_mask` return type** — Task 11's `apply_mask_in_place`
>    treats the mask as `&[u8]`. llguidance 1.7's actual type is likely
>    `SimpleVob` or `Vec<u32>` (packed bitmask). Read llguidance docs, update
>    the helper to use the actual bit-test method (e.g.
>    `mask.is_allowed(token_id)`).
> 2. **`ParserFactory` construction** — replace the `todo!()` body above with
>    real code. Without this, every integration test that calls `Engine::run`
>    panics on first call.
>
> Both fixes are 5–20 lines. They land naturally in Task 13's first commit;
> the plan has them as explicit follow-ups so they don't get missed.

- [ ] **Step 2: Build + test**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo build
cargo clippy --all-targets -- -D warnings
```

- [ ] **Step 3: Update lib.rs**

```rust
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use engine::Engine;

#[cfg(feature = "inference")]
mod engine;
#[cfg(feature = "inference")]
mod generate;
```

- [ ] **Step 4: Commit**

```bash
git add src/engine.rs src/lib.rs
git commit -m "feat(engine): public Engine with from_files/bundled/from_ort_sessions

Per spec §6.2:
- from_files (wasm-incompat) / bundled (uses BUNDLED_TOKENIZER) /
  from_ort_sessions / from_ort_sessions_with_options (wasm-compat)
- generate / generate_with: free-form text generation
- run / run_with: structured output via Task trait + lazy ParserFactory
  (built on first run; cached for subsequent calls)
- warmup: 1024×1024 dummy + 4-token gen (multi-tile path = production
  shape per spec §6.2 doc-comment)
- All public methods carry tracing::instrument spans

Engine: Send + !Sync (per ort::Session). Builds wired through
runtime::{vision,embed_tokens,decoder} validators at construction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 14: task.rs + scene.rs + lib.rs final

**Files:**
- Create: `lfm/src/task.rs` (re-exports of vlm_tasks::*)
- Create: `lfm/src/scene.rs` (lfm-specific SceneTask, parser ported from qwen)
- Modify: `lfm/src/lib.rs` (final form with all re-exports + BUNDLED_*)

- [ ] **Step 1: Write task.rs (trivial re-export)**

```rust
//! Re-exports of the cross-engine `Task` + `ParseError` from `vlm_tasks`.

pub use vlm_tasks::{ParseError, Task};
```

- [ ] **Step 2: Port scene.rs from qwen**

Copy `qwen/src/scene.rs` → `lfm/src/scene.rs` and adjust imports:
- `use vlm_tasks::{ParseError, SceneAnalysis, Task};` (instead of `crate::task::*`)
- Implement `Task for SceneTask` exactly as qwen does
- Same SCENE_PROMPT, same JSON schema, same `DetectionLabels`/`TagList`/`deserialize_*` helpers
- Same indexable-content gate predicate
- All parser tests port verbatim

Then verify all tests pass:

```bash
cargo test --lib scene::
```

- [ ] **Step 3: Write the final lib.rs**

```rust
#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes, missing_docs)]

pub mod chat_template;
pub mod error;
pub mod options;
pub mod preproc;
pub mod scene;
pub mod task;

#[cfg(feature = "inference")]
mod engine;
#[cfg(feature = "inference")]
mod generate;
#[cfg(feature = "inference")]
mod runtime;

pub use error::{Error, Result};
pub use options::{ImageBudget, Options, RequestOptions, ThreadOptions};
pub use preproc::{PreprocessedImage, Preprocessor, TileGrid};
pub use scene::SceneTask;
pub use task::{ParseError, Task};
pub use vlm_tasks::SceneAnalysis;

pub use chat_template::{
  BOS, BOS_TOKEN_ID, EOS_TOKEN_ID, IMAGE_END, IMAGE_START, IMAGE_TOKEN, IMAGE_TOKEN_ID,
  IMAGE_THUMBNAIL, IM_END, IM_START, PAD, PAD_TOKEN_ID,
  TOOL_CALL_END, TOOL_CALL_START,
  expand_image_placeholders,
  BUNDLED_CHAT_TEMPLATE_JINJA,
};

#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use chat_template::{apply_chat_template, ContentItem, Message, UserContent};

#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use engine::Engine;

#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use options::GraphOptimizationLevel;

#[cfg(feature = "decoders")]
pub use preproc::decode_bytes_with_orientation;

#[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
pub use preproc::decode_with_orientation;

/// Bundled `tokenizer.json` (4.5 MB). Used by [`Engine::bundled`].
#[cfg(feature = "bundled")]
#[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
pub const BUNDLED_TOKENIZER: &[u8] = include_bytes!("../models/tokenizer.json");
```

- [ ] **Step 4: Verify all builds**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo build --no-default-features
cargo build --no-default-features --features inference
cargo build  # default = inference + bundled + decoders
cargo test --lib
cargo clippy --all-targets -- -D warnings
```

- [ ] **Step 5: Commit**

```bash
git add src/task.rs src/scene.rs src/lib.rs
git commit -m "feat(public-api): SceneTask + lib.rs final form

- src/task.rs: re-exports vlm_tasks::{Task, ParseError}.
- src/scene.rs: port of qwen::scene::SceneTask with the same prompt,
  schema, and parser fallback (DetectionLabels, TagList, indexable-
  content gate). Tuning per the 450M's drift patterns lands here later.
- src/lib.rs: final re-export surface + BUNDLED_TOKENIZER const.

All builds clean: --no-default-features, --features inference, default.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 15: examples (4) + tests/integration.rs + benches (3)

**Files:**
- Create: `lfm/examples/{smoke,scene_analysis,preprocess_only,qwen_compare}.rs`
- Create: `lfm/tests/integration.rs`
- Create: `lfm/benches/bench_{preproc,tile_grid,chat_template}.rs`

- [ ] **Step 1: examples/smoke.rs**

```rust
//! Phase-zero "does it work" — load the model and run one generation.
//!
//! Usage: `cargo run --release --example smoke -- /path/to/lfm-model-onnx /path/to/image.jpg`

fn main() -> lfm::Result<()> {
  let args: Vec<String> = std::env::args().collect();
  if args.len() != 3 { eprintln!("usage: smoke <model_dir> <image.jpg>"); std::process::exit(1); }
  let model_dir = std::path::Path::new(&args[1]);
  let img_path = std::path::Path::new(&args[2]);

  let engine = lfm::Engine::from_files(
    &model_dir.join("onnx").join("vision_encoder_fp16.onnx"),
    &model_dir.join("onnx").join("embed_tokens_fp16.onnx"),
    &model_dir.join("onnx").join("decoder_model_merged_q4.onnx"),
    &model_dir.join("tokenizer.json"),
    lfm::Options::new(),
  )?;
  let mut engine = engine;
  engine.warmup()?;

  let img = lfm::decode_with_orientation(img_path)?;
  let text = engine.generate(&[img], "Describe this image briefly.")?;
  println!("{text}");
  Ok(())
}
```

- [ ] **Step 2: examples/scene_analysis.rs**

```rust
//! Run SceneTask against one or more keyframes; pretty-print the SceneAnalysis.

fn main() -> lfm::Result<()> {
  let args: Vec<String> = std::env::args().collect();
  if args.len() < 3 { eprintln!("usage: scene_analysis <model_dir> <img1> [img2] ..."); std::process::exit(1); }
  let model_dir = std::path::Path::new(&args[1]);
  let mut engine = lfm::Engine::from_files(
    &model_dir.join("onnx").join("vision_encoder_fp16.onnx"),
    &model_dir.join("onnx").join("embed_tokens_fp16.onnx"),
    &model_dir.join("onnx").join("decoder_model_merged_q4.onnx"),
    &model_dir.join("tokenizer.json"),
    lfm::Options::new(),
  )?;
  let images: Vec<image::DynamicImage> = args[2..].iter()
    .map(|p| lfm::decode_with_orientation(std::path::Path::new(p)))
    .collect::<lfm::Result<_>>()?;
  let task = lfm::SceneTask::new();
  let result = engine.run(&task, &images)?;
  println!("{result:#?}");
  Ok(())
}
```

- [ ] **Step 3: examples/preprocess_only.rs (wasm-compat showcase)**

```rust
//! Wasm-compat showcase: only uses Preprocessor, no Engine.

fn main() -> lfm::Result<()> {
  let path = std::env::args().nth(1).expect("usage: preprocess_only <image>");
  let img = image::open(&path).expect("decode");
  let pre = lfm::Preprocessor::new(lfm::ImageBudget::new());
  let out = pre.preprocess(&img)?;
  println!("rows={} cols={} num_tiles={} num_image_tokens={}",
    out.rows(), out.cols(), out.num_tiles(), out.num_image_tokens());
  println!("pixel_values len = {} (= batch_size {} * patches_per_entry {} * 768)",
    out.pixel_values().len(), out.batch_size(), out.patches_per_entry());
  Ok(())
}
```

- [ ] **Step 4: examples/qwen_compare.rs**

Run both `qwen::Engine` and `lfm::Engine` on the same keyframes; print SceneAnalysis side-by-side. (Requires both `qwen` and `lfm` features. Will need a `[dependencies] qwen = { path = "../qwen" }` added under a `comparison` feature flag in Cargo.toml — opt-in.)

```rust
#[cfg(feature = "comparison")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Load both engines, run SceneTask on the same keyframes, print SceneAnalysis side-by-side.
  // Implementer fills in argv parsing + both engine loads.
  todo!("comparison example — see spec §12.4")
}

#[cfg(not(feature = "comparison"))]
fn main() {
  eprintln!("Run with --features comparison");
}
```

- [ ] **Step 5: tests/integration.rs**

```rust
//! Integration tests gated on `integration` feature + LFM_MODEL_PATH env var.
//! Run with: `LFM_MODEL_PATH=/path/to/model cargo test --features integration --test integration -- --test-threads=1`

#![cfg(feature = "integration")]

use std::path::PathBuf;

fn model_path() -> Option<PathBuf> {
  std::env::var("LFM_MODEL_PATH").ok().map(PathBuf::from)
}

fn build_engine() -> lfm::Engine {
  build_engine_with(lfm::Options::new())
}

/// Bit-stable engine for `deterministic_run_is_idempotent`. Pins
/// ThreadOptions::deterministic() (intra=1, inter=1) on top of the
/// greedy sampler. Without this, ort multi-threaded reductions
/// produce non-deterministic output across runs even with the
/// greedy sampler. See spec §6.3 RequestOptions::deterministic
/// "Bit-stability caveat".
fn build_engine_deterministic() -> lfm::Engine {
  build_engine_with(
    lfm::Options::new().with_thread(lfm::ThreadOptions::deterministic()),
  )
}

fn build_engine_with(opts: lfm::Options) -> lfm::Engine {
  let path = match model_path() {
    Some(p) => p,
    None => { eprintln!("LFM_MODEL_PATH unset, skipping"); std::process::exit(0); }
  };
  lfm::Engine::from_files(
    &path.join("onnx").join("vision_encoder_fp16.onnx"),
    &path.join("onnx").join("embed_tokens_fp16.onnx"),
    &path.join("onnx").join("decoder_model_merged_q4.onnx"),
    &path.join("tokenizer.json"),
    opts,
  ).expect("engine load")
}

#[test]
fn smoke_generate_text_only() {
  let mut e = build_engine();
  let out = e.generate(&[], "What is 2+2?").expect("generate");
  assert!(!out.is_empty());
}

#[test]
fn smoke_generate_one_image() {
  let mut e = build_engine();
  let img = image::open("tests/fixtures/airport_01.jpg").unwrap();
  let out = e.generate(&[img], "Describe this image briefly.").expect("generate");
  assert!(!out.is_empty());
}

#[test]
fn structured_scene_task() {
  let mut e = build_engine();
  let img1 = image::open("tests/fixtures/airport_01.jpg").unwrap();
  let img2 = image::open("tests/fixtures/airport_02.jpg").unwrap();
  let task = lfm::SceneTask::new();
  let result = e.run(&task, &[img1, img2]).expect("run");
  assert!(!result.description().is_empty());
  assert!(result.tags().len() >= 1);
}

#[test]
fn deterministic_run_is_idempotent() {
  // Use the bit-stable build (greedy + intra=1, inter=1).
  let mut e = build_engine_deterministic();
  let img = image::open("tests/fixtures/airport_01.jpg").unwrap();
  let task = lfm::SceneTask::new();
  let a = e.run(&task, &[img.clone()]).expect("run a");
  let b = e.run(&task, &[img]).expect("run b");
  assert_eq!(a, b, "deterministic preset must be bit-stable");
}

#[test]
fn max_tokens_cap_returns_max_tokens_exceeded() {
  let mut e = build_engine();
  let opts = lfm::RequestOptions::deterministic().with_max_new_tokens(3);
  let img = image::open("tests/fixtures/airport_01.jpg").unwrap();
  let task = lfm::SceneTask::new();
  let r = e.run_with(&task, &[img], &opts);
  assert!(matches!(r, Err(lfm::Error::MaxTokensExceeded { max: 3, .. }) | Err(lfm::Error::Parse(_))));
}

#[test]
fn schema_stops_at_closing_brace_not_max_tokens() {
  // Per spec §8.6: SceneTask::schema() must produce a grammar whose
  // is_stopped() fires at the closing brace, NOT trigger max_new_tokens.
  let mut e = build_engine();
  let img = image::open("tests/fixtures/airport_01.jpg").unwrap();
  let opts = lfm::RequestOptions::deterministic().with_max_new_tokens(2048);
  let task = lfm::SceneTask::new();
  let result = e.run_with(&task, &[img], &opts).expect("schema must complete well below cap");
  assert!(!result.description().is_empty());
}

#[test]
fn over_constrained_schema_returns_dead_end() {
  // Spec §12.2: feed an over-constrained schema and verify the
  // sampler surfaces Error::LlGuidanceDeadEnd rather than hanging.
  // We construct a deliberately-impossible Task: schema requires a
  // string field starting with a token combination llguidance can
  // prove no model token sequence can produce, then run().
  use serde_json::json;
  use vlm_tasks::{ParseError, Task};

  struct ImpossibleTask { schema: serde_json::Value }
  impl Task for ImpossibleTask {
    type Output = ();
    fn prompt(&self) -> &str { "Output a sentinel value." }
    fn schema(&self) -> &serde_json::Value { &self.schema }
    fn parse(&self, _raw: &str) -> Result<(), ParseError> { Ok(()) }
  }

  let task = ImpossibleTask {
    // Pattern that the tokenizer cannot match with any token sequence —
    // a literal byte sequence outside the vocab. Concrete construction
    // depends on llguidance 1.7 grammar features; this test may need
    // tuning at impl time once the actual ParserFactory API is wired up.
    schema: json!({
      "type": "object",
      "properties": { "x": { "type": "string", "pattern": "^\\xff\\xff\\xff$" } },
      "required": ["x"],
      "additionalProperties": false,
    }),
  };
  let mut e = build_engine();
  let img = image::open("tests/fixtures/airport_01.jpg").unwrap();
  let r = e.run(&task, &[img]);
  assert!(
    matches!(r, Err(lfm::Error::LlGuidanceDeadEnd { .. })),
    "expected LlGuidanceDeadEnd, got {r:?}"
  );
}

#[test]
fn engine_send_compile_check() {
  fn req<T: Send>() {}
  req::<lfm::Engine>();
  req::<lfm::Preprocessor>();
  req::<lfm::SceneTask>();
}
```

- [ ] **Step 6: benches**

`benches/bench_preproc.rs` — Criterion bench wrapping `Preprocessor::preprocess` on a 1024×1024 dummy.
`benches/bench_tile_grid.rs` — bench `pick_tile_grid` over 100 random sizes.
`benches/bench_chat_template.rs` — bench `apply_chat_template` on a representative message set.

- [ ] **Step 7: Commit**

```bash
git add examples/ tests/integration.rs benches/
git commit -m "feat(examples+tests+benches): smoke, scene_analysis, integration suite, criterion benches

Per spec §12.2 + §12.4 + §12.5:
- examples/smoke.rs: minimal end-to-end
- examples/scene_analysis.rs: run SceneTask, print SceneAnalysis
- examples/preprocess_only.rs: wasm-compat showcase (Preprocessor only)
- examples/qwen_compare.rs: cross-engine comparison (gated on comparison feature)
- tests/integration.rs: 7 tests gated on integration feature + LFM_MODEL_PATH:
  smoke text-only, smoke one image, structured scene task, deterministic
  idempotency, max_tokens cap, schema-stops invariant, Engine: Send check
- benches: criterion harnesses for preproc, tile-grid, chat_template

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

### Task 16: CI workflow + README + CHANGELOG

**Files:**
- Create: `lfm/.github/workflows/ci.yml`
- Create: `lfm/README.md`
- Modify: `lfm/CHANGELOG.md`

- [ ] **Step 1: CI workflow**

`lfm/.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push: { branches: [main, '0.*'] }
  pull_request:
jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build --all-targets
      - run: cargo build --no-default-features --features inference
      - run: cargo test --lib
      - run: cargo clippy --all-targets -- -D warnings
      - run: cargo fmt --check
  wasm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { targets: wasm32-unknown-unknown }
      - run: cargo check --target wasm32-unknown-unknown --no-default-features
      - run: cargo check --target wasm32-unknown-unknown --no-default-features --features decoders
  feature-powerset:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: taiki-e/install-action@cargo-hack
      - run: cargo hack check --feature-powerset --exclude-features cuda,tensorrt,directml,rocm,coreml,integration
```

- [ ] **Step 2: README.md**

Write a concise README:
- One-paragraph overview
- Quick example (load engine, run SceneTask)
- Feature flags table
- License section: dual MIT/Apache for code, LFM Open License v1.0 for weights — link to <https://www.liquid.ai/lfm-license>
- Phase 0 verification flow + script invocation (point to `scripts/README.md`)

- [ ] **Step 3: CHANGELOG.md**

```markdown
# Changelog

## [Unreleased]

### Added
- Initial implementation of `lfm` crate per spec
  `docs/superpowers/specs/2026-05-03-lfm-vlm-wrapper-design.md`.
- `Engine` (sync, raw ort) with `from_files` / `bundled` / `from_ort_sessions{_with_options}`.
- `Engine::generate` / `generate_with` for free-form generation.
- `Engine::run` / `run_with<T: Task>` for structured output via `llguidance`.
- `SceneTask` with the same prompt, schema, and parser as `qwen::SceneTask` (porting v1 — tuning per the 450M's drift patterns deferred).
- Pre-patchified vision encoder input + per-image vision encoder calls (Phase 0 G6 contract).
- Bundled tokenizer.json (default-on `bundled` feature) + `BUNDLED_CHAT_TEMPLATE_JINJA`.
- 7 integration tests gated on `integration` feature.
- Cross-platform CI (Linux x86, macOS aarch64, Windows x86) + wasm subset check.
```

- [ ] **Step 4: Final verification**

```bash
cd /Users/user/Develop/findit-studio/lfm
cargo build --no-default-features
cargo build --no-default-features --features inference
cargo build --no-default-features --features decoders  # wasm subset
cargo build  # default
cargo test --lib
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

All clean.

- [ ] **Step 5: Commit**

```bash
git add .github/ README.md CHANGELOG.md
git commit -m "ci+docs: GitHub Actions matrix + README + CHANGELOG

CI: cross-platform (Linux x86, macOS aarch64, Windows x86) build/test/clippy
+ wasm subset check (--no-default-features [+decoders]) + cargo-hack
feature powerset (excluding EP-vendor-SDK features and integration).

README: usage example + feature flags table + dual-license boundary
(MIT/Apache code vs LFM Open License v1.0 weights).

CHANGELOG: initial release entry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"
```

---

## Self-Review

After completing all 16 tasks, verify against the spec sections:

| Spec section | Implemented in |
|---|---|
| §1 Goals (structured + free-form + wasm preproc) | Tasks 11, 13, 7 |
| §2 Non-goals (no async, no streaming, no tool calling, no continuous batching) | NOT IMPLEMENTED — by design |
| §3 Background | N/A — informational |
| §4 High-level architecture | Tasks 3-14 |
| §5 Workspace + crate layout | Tasks 1, 2, 3 |
| §6.1 Top-level re-exports | Task 14 lib.rs |
| §6.2 Engine | Task 13 |
| §6.3 Options + RequestOptions + ImageBudget + ThreadOptions | Task 5 |
| §6.4 Preprocessor + PreprocessedImage + EXIF helpers | Task 7 |
| §6.5 SceneTask | Task 14 |
| §6.6 Task trait re-exports | Tasks 1, 14 |
| §7 Data flow (preprocess → encode → embed-merge → decode → parse) | Task 12 |
| §7.5 Multi-image vision encoder contract | Task 12 (per-image loop) |
| §8.1 KvCache | Task 10 |
| §8.2 Sampler | Task 11 |
| §8.3 Tile-grid | Task 7 |
| §8.4 `<image>` expansion | Task 6 |
| §8.5 ONNX session validation | Task 8 |
| §8.6 llguidance integration | Tasks 11, 13 |
| §9 Error taxonomy | Task 4 |
| §10 Streaming + tool calling deferral | NOT IMPLEMENTED — by design |
| §11 Bundling strategy | Tasks 3, 6, 14 |
| §12 Testing strategy | Tasks 4-15 + 15 (integration) |
| §13 Open questions / risks / verification gates | Phase 0 fixtures (already in repo) + spec §13 documents the rest |
| §14 Model facts cheat sheet | Reference only |
| §15 Model weights license | Task 13 (Engine doc-comment) + Task 16 (README) |

If any cell is empty, return to the relevant task and add the missing piece.

**Placeholder scan:** two intentional `todo!()` macros remain:
1. **Task 13 `build_parser_factory`** — the exact llguidance 1.7 ParserFactory construction. Surrounded by a 25-line comment block listing exactly what the implementer needs from the docs (TokEnv, ParserFactory, schema entry point, Constraint::compute_mask return type). Must be replaced before Task 15 integration tests can run.
2. **Task 15 `examples/qwen_compare.rs`** — example body, gated behind `feature = "comparison"`. Optional cross-engine A/B example; the body is straightforward (load both engines, run SceneTask on the same fixture set, print side-by-side) but requires both `qwen` and `lfm` working — defer until both are operational.

No other TBDs / TODOs / "implement later" / "similar to Task N" patterns.

**Type consistency check:** spot-checked the following names appear consistently across tasks:
- `RequestOptions` (Tasks 5, 11, 12, 13)
- `ImageBudget` (Tasks 5, 7, 13)
- `PreprocessedImage` (Tasks 7, 12, 13)
- `Sampler` trait + `FreeSampler` + `ConstrainedSampler` (Tasks 11, 12)
- `KvCache` (Tasks 10, 12)
- `IMAGE_TOKEN_ID` (Tasks 6, 12)

All consistent.

---

## v0.1 deferrals (explicitly NOT implemented in this plan)

The plan implements v0 per spec. The following items are noted in the spec as v0.1+ scope; they don't block v0 and don't need plan tasks:

- **Spec §13.2 #21 (Engine-internal preprocess scratch reuse).** Engine reuses the `Vec<PreprocessedImage>` scratch field across calls to avoid ~15 MB allocation per request. Defer to v0.1 — v0 allocates fresh per call, accepting the cost. Add only when an indexing-pipeline benchmark shows it matters.
- **Spec §13.2 #22 (Engine panic-safety contract — "discard on Err mid-generation").** Document on the `Engine` struct that mid-generation panic / Err leaves the Engine in indeterminate state — caller must discard and reconstruct rather than retry. Add as a doc-comment to `Engine` in Task 13. Implementation hardening (clean-on-error) is v0.1+.
- **Spec §13.4 fixture-freshness loud-failure contract.** `validate_*_session` should cite `_metadata.hf_revision` from `tests/fixtures/onnx_io_contract.json` in its error message. v0 produces the generic `Error::SessionShapeMismatch` without fixture-revision context. Either build.rs reads the fixture and emits constants, or v0.1 wires it. Mark as TODO in `validate_*_session` doc-comments.
- **§7.5 batched re-test on future ONNX re-exports.** If LiquidAI re-exports the model, re-run Phase 0 Gate B Case 2 + 3. If `all_passed: true` flips back to true, Engine can switch to a single batched vision call (current per-image discipline becomes unnecessary). Defer; not v0 scope.

**Plan-execution risk note (P0-2 + P1-3 + P2-3):** the highest-risk dependencies are llguidance-related. Both `Constraint::compute_mask` return type (Task 11) and `ParserFactory` construction (Task 13) are pseudocode pinned against expected-but-unverified llguidance 1.7 APIs. The implementer's first move on Task 11 + Task 13 is to read the current llguidance docs and replace these surfaces. Without that, the integration tests in Task 15 panic at first call. Plan flags both inline.

---

**Plan complete and saved to `lfm/docs/superpowers/plans/2026-05-03-lfm-vlm-wrapper.md`.**
