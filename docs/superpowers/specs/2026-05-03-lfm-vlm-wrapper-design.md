# `lfm` — LFM2.5-VL ONNX wrapper — design

**Date:** 2026-05-03 (revised after adversarial review)
**Status:** Approved with documented gaps (see §13 for residuals to verify at impl time)
**Crate:** `lfm` (currently a `template-rs` skeleton at `lfm/`)
**Wraps:** [LiquidAI/LFM2.5-VL-450M-ONNX](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-ONNX)

**Revision history:**
- 2026-05-03 (initial) — first draft, six brainstorming sections approved
- 2026-05-03 (post-review-1) — fixed factual errors (head_dim, llguidance API + version + wasm story, image processor file name), added per-tile position-token detail, expanded §13 with architectural pushbacks and gaps surfaced by adversarial review
- 2026-05-03 (post-review-2) — applied 7 P0 fixes from external review: cache-init asymmetry (conv-state fixed-shape vs KV length-0), `Options::optimization_level` field added, `#[non_exhaustive]` on `Error`, wasm cfg gate on `preprocess_path`, EXIF orientation honored in `preprocess_path`, model-weights license surfaced as new §15. Also took several P1 items (sampling-defaults source citation, deterministic() bit-stability caveat, tool-calling text-only note, OCR limitation in SceneTask, Send/Sync compile-time assertion, exact `"conv"`/`"full_attention"` literals). §13.4 elevated from "during impl" to a Phase 0 verification gate with a runnable `capture_onnx_io.py` script. Did NOT change `min_image_tokens=64` — verified against ground-truth `preprocessor_config.json` (the reviewer's "32" claim came from model-card example text, not the runtime config).
- 2026-05-03 (post-review-3) — extended audit caught a **regression I introduced in post-review-1**: §8.3 had per-tile `smart_resize` in the multi-tile path, which is wrong (upstream uses uniform 512×512 tiles in multi-tile, dynamically-sized only in single-tile path; the thumbnail is dynamically sized). §8.3 rewritten correctly. PreprocessedImage accessors expanded to expose `rows()/cols()/main_tile_size()/thumbnail_size()/tokens_per_main_tile()/thumbnail_tokens()` so `expand_image_placeholders` has the data it needs. §7 step 5 now states the multi-image embed-merge ordering invariant (catches a vision_encoder that batches by tile-index across images). §7 step 6 marks `position_ids` speculative pending §13.4 G1; §8.5 `validate_decoder_session` now treats `position_ids` as conditionally present. §8.6 wording fixed (lazy ParserFactory init, was contradicting itself). Added `is_stopped()` semantics caveat. Trimmed `BUNDLED_*_JSON` from 7 constants to 3 useful ones. Added `from_ort_sessions_with_options` for parity with siglip2/egemma. Added §13.3 entries for special-tokens completeness, accept_empty predicate doc, workspace-lints verification. Did NOT add ImageView<u8> wasm-friendly accessor (deferred to v0.1 per cost/benefit), did NOT add cancellation callback (sync API; caller-side thread+flag works), did NOT change min_image_tokens=64 (re-re-verified — reviewer kept misreading).
- 2026-05-03 (post-review-4) — final adversarial review caught **two new P0s I introduced in post-review-2/3**: (B-1) the proposed `apply_exif_orientation(&mut DynamicImage, &Path)` helper is a phantom API — by the time you have a `DynamicImage` the EXIF metadata is gone. Replaced with `decode_with_orientation(&Path) -> DynamicImage` and `decode_bytes_with_orientation(&[u8]) -> DynamicImage` matching siglip2's pattern. (B-2) `pixel_values` shape was undefined in the multi-tile + thumbnail case — added explicit pad-to-image-max layout invariant + `image_max_size()` accessor + attention-mask-marks-padded-patches contract. Plus: fixed `Options::new()` default to `GraphOptimizationLevel::Level1` (sibling-crate parity, bit-stability rationale); split §13.4 verification into Gate A (metadata, ~30s) and Gate B (multi-image runtime, ~5 min) since G6 needs actual inference; warmup dummy upgraded from 512×512 (single-tile path) to 1024×1024 (multi-tile path = production shape); §11 bundling table synced with §6.1 trim, four unused JSON files dropped from `models/` entirely; added `Engine::generate` blocking-latency warning; §13.3 #14 framing fixed (`max_image_tokens` does NOT bound multi-tile total — flagged as a v0.1 rename candidate). Did NOT change `min_image_tokens` (still 64; same as before).
- 2026-05-03 (post-review-5) — micro-issues from final audit, no P0/P1 left:
  - **C-1**: `image_max_size` now correctly handles thumbnail-exceeds-512 case (wide sources like 1920×1080 produce thumbs ≈ 672×384 — main tiles need width-padding to 672, not assumed 512).
  - **C-2**: CI matrix adds `--target wasm32 --features decoders` check to validate the "wasm-compatible" claim on `decode_bytes_with_orientation`; fallback gating documented if the check fails.
  - **C-3**: Gate B extended with a 1024×1024 multi-tile case (the 256×256 single-tile case alone misses interleaved-tile-row failures).
  - **C-4**: §13.4 explicitly notes Gate A precedes Gate B (Gate B uses the output tensor name Gate A resolves).
  - **C-5**: warmup doc-comment now states the 2–5 second cost up front.
  - **D-1/D-2**: cosmetic plurality + bundled-vs-always gating clarified.
  - **F-1**: `ImageBudget::new()` doc-comment now explicitly explains the `max_image_tokens` asymmetry across paths.
  - **F-2**: §13.4 specifies the fixture-mismatch loud-failure contract.
  - **F-3**: `Sampler::sample` doc clarifies the per-step scratch buffer semantics.
- 2026-05-03 (post-review-6) — final wording-sync nits, no functional changes:
  - **N-1**: §7 step 1 NOTE now matches §6.4 LAYOUT INVARIANT (was describing only the COMMON case; added wide/tall-source caveat with pointer to §6.4).
  - **N-2**: §13.4 G6 prose now describes both the 256×256 and 1024×1024 cases (matches Gate B's two-case script design).
  - **N-3**: §13.4 reorganized — Gate A (metadata) and Gate B (runtime) now have separate fact-resolution blocks (G6 was previously listed under "metadata gate resolves" which was wrong).
  - **N-4**: Gate B `fp16_tol` pinned at `5e-3` with rationale (was undefined in pseudocode).
  - **N-5**: §11 tarball estimate revised to ~1.8–2.2 MB (was unrealistically low at 1.5 MB given the bundled tokenizer alone compresses to ~1.7–2.0 MB).
- 2026-05-03 (post-review-7) — deep-examination round caught 3 P2 + 11 P3 in areas earlier rounds skimmed (type signatures, panic-safety, observability, scratch buffers, verification-script metadata):
  - **P2-1**: `KvCache` had `HashMap<&'static str, Tensor>` — but ort returns borrowed `&str` from the session, not `'static`. Rust type error. Changed to `HashMap<SmolStr, Tensor>` + a `present_to_past: HashMap<SmolStr, SmolStr>` name map (also runtime-discovered).
  - **P2-2**: Gate B added Case 3 (256² + 1024² mixed-size cross-batch). Cases 1+2 both have identical image_max within the batch and don't exercise cross-batch repadding; Case 3 forces it. Real failure mode (encoder bleeding attention from larger image into smaller image's padded region) that 1+2 miss.
  - **P2-3**: `capture_onnx_io.py` now writes `_metadata` block with `captured_at`, `hf_repo`, `hf_revision`, `capture_script_version`. The fixture-freshness loud-failure contract (post-review-5 F-2) referenced metadata fields the script wasn't producing. Now anchored.
  - **P3-1**: Stop conditions in §7 step 6 expanded with edge cases (max_new_tokens=0 rejected at validation, llguidance dead-end vs EOS race, schema-complete + EOS tie-breaker).
  - **P3-2**: §6.2 `generate` doc warns about literal `"<image>"` in user prompt → `Error::ImageTokenCountMismatch`.
  - **P3-3 / 4 / 5 / 8 / 9 / 10 / 11**: Added §13.2 entries 19–25 covering validation rules enumeration, tracing instrumentation levels, scratch-buffer reuse pattern, panic-safety contract, `Engine: Send + !Sync` rationale, `from_ort_sessions` wasm-claim rephrase, `Embedding` public-export intent.
  - **P3-6**: §12.1 added `runtime/session.rs` test row for `check_outlet` rejection cases.
  - **P3-7**: §12.2 added `schema_stops_at_closing_brace_not_max_tokens` integration test pinning §8.6's contract.
- 2026-05-03 (post-review-8, **final**) — round-7 audit verdict: "ship it; round 8 has near-zero marginal value." Took only the highest-value items (operator-facing or genuinely-misleading-wording), skipped the rest per reviewer's explicit guidance:
  - **C-4**: cumulative startup latency (3–8 s blocking) called out in `warmup` doc — service operators sizing readiness probes were getting only the per-method numbers.
  - **C-5**: `ParserFactory` build cost (0.1–2 s) noted in §8.6 — first `run` call has a one-time grammar-compile tax that was previously invisible.
  - **C-7**: §13.2 #21 wording disambiguated — the Engine-internal scratch reuse is NOT a public `Preprocessor` signature change.
  - **D-1**: §13.4 #18 added — fixture-maintenance cadence (manual in v0; cron-bot for v0.1+).
  - **D-3**: §13.1 #6 added — production observability (metrics, health-check, token counters) explicitly v0.1+ scope.
  - **D-4**: §12.2 added `--test-threads=1` requirement with rationale (ort runtime is process-global; concurrent Engine constructions race).
  - **D-6**: §14 added memory-profile rows (idle ~770 MB, per-call peak, startup latency).
  - **D-7**: §12.6 added cross-platform OS/arch matrix (Linux x86, macOS aarch64, Windows x86; Linux aarch64 deferred).
  - SKIPPED per reviewer guidance: C-1 (numbering — leave as-is), C-2 (re-export header — minor), C-3 (pseudocode style), C-6 (`Source` trait — v0.1 polish), D-2 (qwen sequencing — writing-plans concern), D-5 (`cargo publish --dry-run` gate — writing-plans concern).
- **Status: SPEC SEALED.** Eight rounds of adversarial review (77 cumulative findings, 76 closed + 1 retraction) caught ten distinct correctness bugs that would each have produced confidently-wrong code at integration time (head_dim error, phantom EXIF helper, multi-tile per-tile smart_resize regression, cache-init asymmetry, `position_ids` over-confidence, `image_max` wide-source case, `KvCache` `'static` lifetime error, layout-invariant cross-section drift, missing cross-batch repadding test, fixture-freshness metadata gap).
- 2026-05-03 (post-Phase-0) — Phase 0 verification gates run successfully; both fixture JSONs committed. Major spec updates from the captured contract:
  - **§6.4 PreprocessedImage rewritten**: vision encoder INPUT is `[batch_size, num_patches, 768]` pre-patchified (NOT `[N, 3, H, W]` image-shaped). Padding is at the patch level via `pixel_attention_mask`, NOT at the pixel level via `image_max_size`. The whole pre-Phase-0 LAYOUT INVARIANT was wrong; the actual semantics are patch-padding. Earlier confusion came from the README's Python sample showing `pixel_values.numpy()` without an explicit shape annotation.
  - **§7 step 4 rewritten**: vision encoder output is rank 2 `[num_image_tokens, 1024]` (NOT rank 3). Per-image vision encoder calls (one per source image), not batched — see §7.5.
  - **§7 step 6 rewritten**: decoder has NO `position_ids` input. Cache uses sparse layer indices `[0,1,3,4,6,7,9,11,13,15]` for conv and `[2,5,8,10,12,14]` for attn (matches `layer_types` raw indices). Output tensor name is `image_features` (§8.5 placeholders resolved).
  - **§7.5 NEW**: multi-image vision encoder contract. Phase 0 Gate B's three test cases proved that batched multi-image calls SILENTLY CORRUPT outputs when any image routes through the multi-tile path (Case 2: max diff 286.875; Case 3: max diff 473.25). Implementation MUST call `vision_encoder.run` once per source image. Future-proof: if a re-export fixes the batched case, fixture's `all_passed` flips and Engine can switch to batched.
  - **§8.1 KvCache table updated**: sparse layer indices documented; G2/G3 marked RESOLVED.
  - **§8.5 validators rewritten**: actual ONNX outlet names (`image_features`, `inputs_embeds`), no `position_ids` check, sparse-index check for cache tensors, pre-patchified `pixel_values` shape `[-1, -1, 768]`.
  - **§13.4 Phase 0 marked COMPLETE**: all six gates resolved; the historical "must run before Rust code" framing preserved as a re-capture protocol for future ONNX re-exports.
  - **Real correctness bug avoided**: the pre-Phase-0 spec would have produced a Rust crate that batches multi-image vision calls and silently emits wrong embeddings for any input larger than ~512×512. Per-image vision calls are the only correct approach for this ONNX export. **This was caught by the Phase 0 gate, not at integration time.** The reviewer's "the marginal value of further review is zero, but the marginal cost of skipping Phase 0 is real" assessment proved exactly right — Phase 0 caught a correctness bug that no amount of spec audit could have surfaced.
- 2026-05-03 (post-Phase-0 round 9 audit) — caught 2 P1 internal-consistency drifts that were missed when the canonical sections were rewritten post-Phase-0:
  - **P1-1 (§6.4 Multi-image flow paragraph)**: still said "concatenates ... before one vision_encoder.run call" — directly contradicting §7.5's per-image contract. Fixed: rewrote step 2 to say "calls vision_encoder.run once per source image (per-image, NOT batched — see §7.5)". Real bug-amplification risk if an implementer read this paragraph and didn't cross-check §7.5.
  - **P1-2 (§13.2 #21 scratch-buffer + §14 cheat sheet memory line)**: described the scratch as `[N_tiles, 3, image_max_h, image_max_w]` (the OLD pre-Phase-0 image-shaped layout). Fixed to `[N_batch, num_patches, 768]` per the actual ONNX contract. Size estimate (~15 MB) coincidentally close — kept as ~15 MB with corrected dimensional reasoning.
  - **D-2 (Gate B pseudocode placeholder)**: `OUT_NAME = "image_embeds"` was forward-looking-correct pre-Phase-0 but actively misleading post-Phase-0 (actual is `image_features`). Fixed to use the resolved name with a comment that future re-runs of Gate A re-derive it.
  - Added §13.4 #18 **post-re-capture audit protocol** — explicitly listing the secondary spec sections to audit after any future Phase 0 re-capture. The round-9 review made the drift risk explicit; this codifies the mitigation.
- **Status: SPEC SEALED FOR WRITING-PLANS INVOCATION.** Nine rounds of adversarial review + Phase 0 verification + this consistency-drift sweep. The post-Phase-0 round caught precisely the secondary-section drift that the reviewer correctly flagged as the only category remaining at risk after Phase 0. Implementation plan can now be authored against contracts grounded in observed reality with no internal contradictions.
- **Next: invoke `superpowers:writing-plans`.**

---

## 1. Goals

`lfm` is a sibling crate to `qwen` / `siglip2` / `egemma` in the findit-studio monorepo. It wraps the LFM2.5-VL-450M ONNX model with a Rust API matching the codebase's existing idioms, supporting:

1. **Structured-output inference** (`Engine::run<T: Task>`) producing typed Rust output with **schema-guaranteed** JSON via `llguidance`-constrained sampling. Same `Task` trait surface as `qwen`'s; `SceneTask`/`SceneAnalysis` parity so the same indexing pipeline can use either model.
2. **Free-form generation** (`Engine::generate`) for VQA / chat / arbitrary text generation — the entry point that uses no constraints.
3. **Wasm-compatible preprocessing subset** (`Preprocessor`) that compiles without `ort` / `tokenizers` / `llguidance`, so in-browser callers can preprocess frames and ship them to a server-side instance.

## 2. Non-goals

- **Async API.** Raw `ort 2.0.0-rc.12` is sync + `Send + !Sync`. Wrapping in `spawn_blocking` would be cosmetic. Sync `&mut self` on the hot path matches the runtime's actual concurrency model and the existing siglip2/egemma idiom.
- **Streaming generation in v0.** Single-shot returns full `String`. Adding a streaming variant later is non-breaking — see §10.
- **Tool / function calling in v0.** The model has `<|tool_call_start|>` tokens and the chat template renders tools, but none of that is wired to public API in v0. See §10.
- **Continuous batching.** Single-instance, single-call. Concurrent callers serialize behind `Mutex<Engine>` if needed.
- **Mixed-model batch APIs.** No `embed_batch`-style entry points (the generative path doesn't naturally batch the way embedding paths do).
- **Cancellation.** Sync API; caller wraps in a thread + cancel-flag if needed.
- **GGUF / safetensors / non-ONNX backends.** ONNX only (3-graph: vision + embed + decoder).

## 3. Background

### 3.1 The model

`Lfm2VlForConditionalGeneration` = SigLIP2-86M vision encoder + 2-layer GELU projector + LFM2-350M hybrid LM. Total 450M params, BF16 weights. ONNX export ships as 3 separate graphs:

```
onnx/
├── vision_encoder{,_fp16,_q4,_q8}.onnx       # post-projector embeddings, 1024-dim
├── embed_tokens{,_fp16}.onnx                 # input_ids → embeddings
└── decoder_model_merged{,_fp16,_q4,_q8}.onnx # autoregressive decoder w/ KV cache
```

Plus `tokenizer.json` (4.5 MB), `chat_template.jinja` (3.8 KB), and `preprocessor_config.json` (0.7 KB) — see §11 for which we bundle vs not.

**LM architecture is hybrid** — 16 layers split as `[conv, conv, attn, conv, conv, attn, conv, conv, attn, conv, attn, conv, attn, conv, attn, conv]`. **10 conv layers + 6 full-attention layers**. Conv layers keep a 3-token state cache (`conv_L_cache=3`). Attention layers use GQA (16 heads, 8 KV heads). This means the decoder has **22 cache tensors per step**: 10 `past_conv.*` + 12 `past_key_values.*.{key,value}`. Non-standard — most decoder wrappers handle KV caches but not conv-state caches.

**Vision** is a literal `siglip2_vision_model` — same family our `siglip2-naflex` crate wraps. 256 patches per 512×512 tile at patch-size 16. The vision encoder's ONNX output is **post-projector** (1024-dim, ready to merge) — the projector MLP (768×4 → 2048 → 1024 with GELU) is fused into `vision_encoder.onnx`.

**Image preprocessing** uses dynamic tile-grid selection with per-call budgeting:
- Tile size 512×512, encoder patch size 16, downsample factor 2 (pixel-shuffle 2×2 → 1 token, so 256 tokens per full tile).
- Caller-tunable budget: `min_image_tokens=64`, `max_image_tokens=256`, `min_tiles=2`, `max_tiles=10`. Optional thumbnail tile for global context.
- Normalization: rescale `1/255`, mean=std=`0.5` → pixels in `[-1, 1]` (NOT ImageNet stats).

**Special tokens** (verified in tokenizer_config.json):
- `<|startoftext|>` (BOS, id 1), `<|im_end|>` (EOS, id 7), `<|pad|>` (id 0).
- ChatML markers: `<|im_start|>`, `<|im_end|>`.
- Image tokens: `<image>` (id **396**), `<|image_start|>`, `<|image_end|>`, `<|img_thumbnail|>`.
- Tool tokens: `<|tool_call_start|>`, `<|tool_call_end|>`.

**Recommended sampling** (model card): `temperature=0.1`, `min_p=0.15`, `repetition_penalty=1.05`. Note `min_p` (NOT `top_p`/`top_k`).

**Native structured-output story:** the model card shows JSON-shaped output via prompt engineering (e.g., bbox detection) and tool-calling via the chat template's `tools` slot, but **no built-in constrained-decoding API**. We add that ourselves via `llguidance`.

### 3.2 The codebase context

Three sibling crates exist:

- **`qwen`** — Qwen3-VL-2B for structured scene-analysis output. Built on `mistralrs` (Metal-only, async). Owns `Task` trait, `SceneTask`, `SceneAnalysis`, and a battle-tested resilient parser.
- **`siglip2`** — image+text bi-encoder for embeddings via raw `ort`. Defines the conventions we follow: feature-gated `inference`, wasm subset, EP gates (cuda/tensorrt/directml/rocm/coreml), strict ONNX session validation, `from_files`/`bundled`/`from_parts` constructor pattern.
- **`egemma`** — text-encoder for embeddings via raw `ort`. Same conventions as siglip2, smaller surface.

`lfm` adopts:
- `siglip2`/`egemma` runtime conventions (raw `ort`, sync, feature gates, EPs, wasm subset, strict session validation, `from_files`/`bundled`/`from_ort_sessions` constructors).
- `qwen` public-API shape (`Engine` + `Task` + structured output), extended with a free-form `generate` method.

## 4. High-level architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ lfm crate                                                        │
│                                                                  │
│  Public API                                                      │
│    Engine ──┬─ generate(images, prompt) -> String               │
│             ├─ run<T: Task>(task, images) -> T::Output          │
│             │   (uses llguidance with task.schema())             │
│             └─ from_files / bundled / from_ort_sessions          │
│    Preprocessor (wasm-compat — no ort/tokenizers/llguidance)    │
│    SceneTask : Task<Output = vlm_tasks::SceneAnalysis>           │
│    Options, RequestOptions, ImageBudget, ThreadOptions           │
│    Error, Result                                                 │
│                                                                  │
│  Internals (Approach 2 module structure)                         │
│    preproc/   tile-grid + normalize (wasm-compat)                │
│    runtime/   3 ort wrappers + KvCache + Sampler                 │
│    chat_template.rs   apply_chat_template + special-token consts │
│    generate.rs        end-to-end pipeline (prompt build,         │
│                       encode, embed-merge, decode loop, decode)  │
│    engine.rs          public Engine                              │
│    scene.rs           lfm-specific SceneTask impl                │
│    task.rs            re-exports of vlm_tasks::Task / ParseError │
└──────────────┬───────────────────────────────────────────────────┘
               │
               │ depends on
               ▼
┌──────────────────────────────────────────────────────────────────┐
│ vlm-tasks crate (NEW)                                            │
│   pub trait Task                                                 │
│   pub enum ParseError                                            │
│   pub struct SceneAnalysis  ← canonical type, used by both       │
│                               qwen and lfm engines               │
└──────────────────────────────────────────────────────────────────┘
               ▲
               │ depends on (small migration)
               │
┌──────────────────────────────────────────────────────────────────┐
│ qwen crate (existing — minor touch)                              │
│   qwen::scene::SceneAnalysis  → re-export of vlm_tasks::SceneAnalysis │
│   qwen::Task / qwen::ParseError → re-exports                     │
│   qwen::scene::SceneTask stays put — just constructs the         │
│     canonical SceneAnalysis instead of its own                   │
└──────────────────────────────────────────────────────────────────┘
```

## 5. Workspace & crate layout

### 5.1 New `vlm-tasks` crate

```
findit-studio/vlm-tasks/
├── Cargo.toml
├── src/
│   ├── lib.rs           # re-exports
│   ├── task.rs          # Task trait + ParseError
│   └── scene.rs         # SceneAnalysis (data type only; accessor surface)
└── README.md
```

**Owns:**
- `Task` trait (lifted verbatim from `qwen::task`).
- `ParseError` (lifted verbatim).
- `SceneAnalysis` (lifted verbatim from `qwen::scene::SceneAnalysis`).
- Accessors for `SceneAnalysis` (the `scene/description/subjects/...` getters and `with_*`/`set_*` builders).

**Does NOT own:**
- `SceneTask` impls — each engine ships its own (qwen's stays in `qwen::scene::SceneTask`; lfm's lives in `lfm::scene::SceneTask`).
- Defensive parser primitives (`DetectionLabels`, `TagList`, `deserialize_optional_*` helpers) — those are tuning details per engine. Each engine duplicates what it needs. The 450M model and the 2B model have different drift patterns; forcing identical parser logic across crates is a maintenance trap.

**Cargo.toml:**

```toml
[package]
name = "vlm-tasks"
version = "0.1.0"
edition = "2024"
rust-version = "1.95"
description = "Shared types for findit-studio VLM engines: Task trait, SceneAnalysis"
license = "MIT OR Apache-2.0"

[dependencies]
serde      = { version = "1", features = ["derive"], optional = true }
serde_json = "1"
smol_str   = "0.3"
thiserror  = "2"

[features]
default = []
serde   = ["dep:serde", "smol_str/serde"]
```

### 5.2 `qwen` migration

Three type definitions move; behavior unchanged. All existing `qwen::*` import paths continue to work via re-exports.

```rust
// qwen/src/lib.rs
pub use vlm_tasks::{ParseError, Task};
pub use vlm_tasks::SceneAnalysis;  // also re-exported below

// qwen/src/scene.rs
pub use vlm_tasks::SceneAnalysis;
// SceneTask impl stays here, unchanged. Its parser machinery
// (DetectionLabels, TagList, etc.) stays here too — qwen-tuned.
```

`qwen/Cargo.toml` adds `vlm-tasks = { path = "../vlm-tasks" }`. No other dep changes. Tests pass without modification.

### 5.3 `lfm` crate

```
findit-studio/lfm/
├── Cargo.toml
├── build.rs                       # no-op (template-rs default); reserved for tokenizer/config validation
├── README.md
├── CHANGELOG.md
├── docs/superpowers/specs/        # this file
├── models/                        # bundled assets (whitelisted by Cargo.toml `include`)
│   ├── tokenizer.json             # 4.5 MB — `feature = "bundled"`, exposed as BUNDLED_TOKENIZER
│   ├── chat_template.jinja        # 3.8 KB — always included, exposed as BUNDLED_CHAT_TEMPLATE_JINJA
│   └── preprocessor_config.json   # 0.7 KB — build-fixture only, NOT exposed (unit test verifies ImageBudget::new() matches)
├── src/
│   ├── lib.rs                     # re-exports + features + bundled consts
│   ├── error.rs                   # Error, Result
│   ├── options.rs                 # Options, RequestOptions, ImageBudget, ThreadOptions
│   ├── chat_template.rs           # apply_chat_template + special-token consts
│   ├── embedding.rs               # post-projector vision-tile embedding type
│   ├── preproc/
│   │   ├── mod.rs                 # Preprocessor + PreprocessedImage (wasm-compat)
│   │   └── tile_grid.rs           # tile-grid algorithm (wasm-compat)
│   ├── runtime/
│   │   ├── mod.rs                 # re-exports
│   │   ├── session.rs             # build_session + validate_*_session
│   │   ├── vision.rs              # VisionEncoder
│   │   ├── embed_tokens.rs        # EmbedTokens
│   │   ├── decoder.rs             # Decoder + KvCache
│   │   └── sampler.rs             # Sampler trait + FreeSampler + ConstrainedSampler
│   ├── generate.rs                # end-to-end pipeline
│   ├── engine.rs                  # public Engine
│   ├── task.rs                    # re-exports of vlm_tasks::Task / ParseError
│   └── scene.rs                   # lfm-specific SceneTask impl
├── examples/
│   ├── smoke.rs                   # minimal "does it work"
│   ├── scene_analysis.rs          # run(SceneTask, keyframes)
│   ├── qwen_compare.rs            # both engines on same fixtures
│   └── preprocess_only.rs         # wasm-compat showcase
├── benches/
│   ├── bench_preproc.rs
│   ├── bench_tile_grid.rs
│   └── bench_chat_template.rs
└── tests/
    ├── integration.rs             # gated on `integration` feature, requires LFM_MODEL_PATH
    └── fixtures/
        ├── airport_01.jpg         # ports from qwen so we can A/B
        ├── airport_02.jpg
        ├── airport_03.jpg
        ├── chat_template_cases.json
        ├── tile_grid_cases.json
        ├── image_expansion_cases.json
        └── scene_payloads/
```

**Cargo.toml:**

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
ort         = { version = "2.0.0-rc.12", optional = true }
tokenizers  = { version = "0.23", optional = true }
llguidance  = { version = "1.7", optional = true }    # current as of 2026-05; tracks 1.x API
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
# Activates the ONNX-backed inference path (Engine + run + generate).
# Pulls ort + tokenizers + llguidance, none of which target wasm32.
inference   = ["dep:ort", "dep:tokenizers", "dep:llguidance"]
# Embeds tokenizer.json (4.5 MB) via include_bytes! and adds
# Engine::bundled constructor.
bundled     = ["inference"]
# Adds JPEG/PNG decoders to the `image` crate. Gates Preprocessor::preprocess_path.
decoders    = ["image/jpeg", "image/png"]
# Pulls serde + serde_json. Activates Serialize/Deserialize on Options,
# RequestOptions, ImageBudget, ThreadOptions.
serde       = ["dep:serde", "dep:serde_json", "smol_str/serde", "vlm-tasks/serde"]

# Opt-in execution providers — same vocabulary siglip2/egemma use.
cuda     = ["inference", "ort/cuda"]
tensorrt = ["inference", "ort/tensorrt"]
directml = ["inference", "ort/directml"]
rocm     = ["inference", "ort/rocm"]
coreml   = ["inference", "ort/coreml"]

# Enables tests/integration.rs (requires LFM_MODEL_PATH env var).
integration = ["inference"]

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
required-features = ["inference"]
[[example]]
name              = "preprocess_only"
# Demonstrates the wasm-compat subset — no `inference` required.

[[bench]]
name    = "bench_preproc"
harness = false
[[bench]]
name    = "bench_tile_grid"
harness = false
[[bench]]
name    = "bench_chat_template"
harness = false

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

[lints]
workspace = true
```

(Workspace lints mirror siglip2/egemma — `rust_2018_idioms`, `single_use_lifetimes`, `unexpected_cfgs` with `docsrs` allowed.)

## 6. Public API

### 6.1 Top-level re-exports (`lib.rs`)

```rust
// === always available (wasm-compatible) ===
pub use error::{Error, Result};
pub use options::{ImageBudget, Options, RequestOptions, ThreadOptions};
pub use preproc::{PreprocessedImage, Preprocessor};
#[cfg(feature = "decoders")]
pub use preproc::decode_bytes_with_orientation;
#[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
pub use preproc::decode_with_orientation;
pub use embedding::Embedding;
pub use chat_template::{
  apply_chat_template,
  IMAGE_TOKEN, IMAGE_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID,
  IM_START, IM_END, IMAGE_START, IMAGE_END, IMAGE_THUMBNAIL,
  TOOL_CALL_START, TOOL_CALL_END,
};
pub use vlm_tasks::{ParseError, SceneAnalysis, Task};
pub use scene::SceneTask;

// === inference-gated ===
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use engine::Engine;
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use options::GraphOptimizationLevel;

// === bundled constants (trimmed to what's actually useful at runtime) ===
//
// Two constants exposed, each with different gating:
// - BUNDLED_TOKENIZER (4.5 MB): bundle-feature-gated because of size.
//   Used by Engine::bundled to construct the Tokenizer in-process
//   without a tokenizer_json path. Callers who don't need the
//   convenience opt out via `--no-default-features --features inference`.
// - BUNDLED_CHAT_TEMPLATE_JINJA (3.8 KB): always shipped. Tiny and
//   useful for parity tests against the Rust port even when inference
//   is off; gating it would be unnecessary surface-area thrash.
//
// preprocessor_config.json ships in models/ but is NOT exposed publicly.
// It's used as a build-time / unit-test fixture only — a unit test parses
// it and asserts ImageBudget::new() matches each field, catching drift
// between Rust constants and the upstream config. No consumer benefit
// from exposing it as a public &str constant (the constants are already
// baked into Rust code).
//
// config.json, generation_config.json, processor_config.json,
// tokenizer_config.json are NOT shipped at all (see §11): every value we
// need from them is already a Rust constant; shipping them as part of the
// .crate is dead weight + LFM-licensed payload no one reads.
#[cfg(feature = "bundled")]
#[cfg_attr(docsrs, doc(cfg(feature = "bundled")))]
pub const BUNDLED_TOKENIZER: &[u8] = include_bytes!("../models/tokenizer.json");
pub const BUNDLED_CHAT_TEMPLATE_JINJA: &str = include_str!("../models/chat_template.jinja");
```

### 6.2 `Engine` (in `engine.rs`)

```rust
pub struct Engine { /* private — 3 sessions + tokenizer + opts */ }

impl Engine {
  // === construction ===

  /// Loads three ONNX sessions + a tokenizer file. Wasm-incompatible
  /// (`ort 2.0.0-rc.12` cfg-gates `commit_from_file` out of wasm32).
  /// Wasm callers use [`Self::from_ort_sessions`].
  #[cfg(not(target_arch = "wasm32"))]
  pub fn from_files(
    vision_onnx: &Path,
    embed_onnx: &Path,
    decoder_onnx: &Path,
    tokenizer_json: &Path,
    opts: Options,
  ) -> Result<Self>;

  /// Loads three ONNX sessions; tokenizer comes from the embedded
  /// `BUNDLED_TOKENIZER` (4.5 MB).
  #[cfg(all(feature = "bundled", not(target_arch = "wasm32")))]
  pub fn bundled(
    vision_onnx: &Path,
    embed_onnx: &Path,
    decoder_onnx: &Path,
    opts: Options,
  ) -> Result<Self>;

  /// Constructs from caller-built sessions + tokenizer with crate-default
  /// `Options::new()`. Wasm-compatible (caller is responsible for building
  /// sessions via wasm-specific `ort` async APIs). Equivalent to
  /// `from_ort_sessions_with_options(..., Options::new())`.
  pub fn from_ort_sessions(
    vision: ort::session::Session,
    embed: ort::session::Session,
    decoder: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
  ) -> Result<Self>;

  /// Same as [`Self::from_ort_sessions`] with caller-supplied [`Options`].
  /// Mirrors siglip2/egemma's `from_ort_session_with_options` pattern.
  pub fn from_ort_sessions_with_options(
    vision: ort::session::Session,
    embed: ort::session::Session,
    decoder: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
    opts: Options,
  ) -> Result<Self>;

  // === accessors ===

  pub const fn options(&self) -> &Options;
  pub const fn request(&self) -> &RequestOptions;
  pub const fn image_budget(&self) -> &ImageBudget;

  // === inference (sync, &mut self) ===

  /// Runs one 4-token generation against a 1024×1024 dummy black image.
  /// JITs graph kernels and primes the KV-cache shape. Logs duration
  /// at `debug`.
  ///
  /// **Cost: 2–5 seconds on CPU.** Warmup runs a real prefill on
  /// ~1024 image tokens + 4 decode steps — significantly more work
  /// than a no-op warm. **Call once at startup, not per-request.**
  /// A caller wrapping `Engine::from_files() + warmup()` in a
  /// "construct on demand" path will hit unexpected first-call latency.
  ///
  /// **Cumulative startup:** `from_files` (~1–3 s for ort to load
  /// + memory-map ONNX weights) + `warmup` (2–5 s) = **3–8 s of
  /// blocking startup** before the first request. Service operators
  /// must size liveness/readiness probes accordingly — defaults of
  /// "ready within 1 s" will time out.
  ///
  /// Dummy size = 1024×1024 specifically to route through the
  /// **multi-tile path** (per §8.3, multi-tile triggers when image
  /// area > `max_pixels_tolerance × max_image_tokens × 32²` =
  /// `2.0 × 256 × 1024 = 524288` pixels; 1024×1024 = 1M pixels >
  /// threshold; 512×512 = 262K pixels < threshold and would route
  /// single-tile). Since production keyframes are typically 720p+
  /// and route multi-tile, warming on a single-tile-path size selects
  /// the wrong GEMM kernels for the production shape. 1024×1024
  /// produces a `(rows × cols, 512, 512)` main-tile batch + a thumbnail
  /// — the same shape envelope a typical production call hits.
  pub fn warmup(&mut self) -> Result<()>;

  /// Free-form text generation. Decodes special tokens off; stops on
  /// EOS or `max_new_tokens`. Empty `images` slice is valid — text-only
  /// generation skips the vision encoder and `<image>` expansion.
  ///
  /// **Latency:** typically 5–30 seconds on CPU depending on
  /// `max_new_tokens` and image complexity. This is a **synchronous,
  /// blocking** call — the calling thread is held for the full
  /// generation. For cancellation, wrap in a dedicated thread + cancel
  /// flag at the call site (we don't support mid-call cancellation;
  /// the sync API would have to drop ort sessions to abort, which
  /// would force a full reload on the next call).
  ///
  /// **Literal `<image>` in `prompt`:** if `prompt` contains the
  /// substring `"<image>"` (e.g. `"What does the <image> tag mean in
  /// HTML?"`), the tokenizer would emit `IMAGE_TOKEN_ID(396)` at that
  /// position, but `image_features` from the vision encoder won't have
  /// a row to merge there. Engine returns
  /// `Error::ImageTokenCountMismatch { expected: images.len(), got: N }`.
  /// Caller's responsibility to escape or rephrase if a literal
  /// `<image>` is intended in the user's text.
  pub fn generate(&mut self, images: &[DynamicImage], prompt: &str) -> Result<String>;

  /// Per-call sampler override.
  pub fn generate_with(
    &mut self,
    images: &[DynamicImage],
    prompt: &str,
    request: &RequestOptions,
  ) -> Result<String>;

  /// Structured output. Uses `llguidance` with `task.schema()` to mask
  /// invalid next-tokens at sampling time → guaranteed schema-valid
  /// JSON → `task.parse(text)` → typed `Output`.
  pub fn run<T: Task>(&mut self, task: &T, images: &[DynamicImage]) -> Result<T::Output>;

  /// Per-call sampler override.
  pub fn run_with<T: Task>(
    &mut self,
    task: &T,
    images: &[DynamicImage],
    request: &RequestOptions,
  ) -> Result<T::Output>;
}
```

**Rationale notes:**

- **`images: &[DynamicImage]` (borrow), not `Vec` (consume).** Different from qwen because we control preprocessing and don't need ownership. Caller can re-use images.
- **Empty-`images` is valid.** Pure text generation works (skip vision encoder, skip `<image>` expansion). Useful for tool-calling later.
- **`warmup` runs one 4-token generation against a 1024×1024 dummy black image.** Dummy size routes through the **multi-tile path** (the production shape; see warmup doc-comment in §6.2 for the threshold math). 1×1 / 512×512 would route single-tile and select the wrong GEMM kernels for production keyframes. The 4-token cap covers the full-prefill decoder shape AND one incremental-decode step — both major shapes the engine encounters.

### 6.3 `Options` (in `options.rs`)

```rust
pub struct Options {
  request: RequestOptions,
  image_budget: ImageBudget,
  thread: ThreadOptions,
  /// Forwarded to ort::session::SessionBuilder::with_optimization_level
  /// at session-build time. Mirrors siglip2/egemma. Gated on `inference`
  /// because GraphOptimizationLevel only exists when ort is present;
  /// the serde feature provides a string-mirror enum for deserializing
  /// from config (so wasm builds without `inference` can still parse
  /// an Options blob to ship to a server).
  #[cfg(feature = "inference")]
  optimization_level: GraphOptimizationLevel,
}

impl Options {
  /// Defaults: `RequestOptions::deterministic()`, `ImageBudget::new()`,
  /// `ThreadOptions::default()`, `GraphOptimizationLevel::Level1`.
  ///
  /// Level1 (not Level3) matches the siglip2/egemma idiom — higher
  /// optimization levels can subtly alter numerics (Level2+ enables
  /// kernel fusions that may reorder reductions; Level3 adds extended
  /// transforms that can affect bit-exact comparison). For a generative
  /// VLM where `RequestOptions::deterministic()` is the engine-level
  /// default, surfacing Level3 implicitly would silently undermine the
  /// bit-stability guarantee. Callers who want maximum throughput and
  /// don't care about bit-stability opt in via
  /// `Options::with_optimization_level(GraphOptimizationLevel::Level3)`.
  pub const fn new() -> Self;

  // builder + in-place setters per field, including:
  #[cfg(feature = "inference")]
  pub const fn with_optimization_level(self, level: GraphOptimizationLevel) -> Self;
  #[cfg(feature = "inference")]
  pub const fn set_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestOptions {
  temperature: f32,         // 0.1 (model card) | 0.0 (deterministic)
  min_p: f32,               // 0.15 — NOTE: LFM uses min_p, not top_p/top_k
  repetition_penalty: f32,  // 1.05
  max_new_tokens: usize,    // 512
}

impl RequestOptions {
  /// Defaults from the LFM2.5-VL-450M model card text (NOT from
  /// `generation_config.json` — that file only carries
  /// bos/eos/pad_token_id and transformers_version, not sampler
  /// hyperparameters). Values: `temperature=0.1`, `min_p=0.15`,
  /// `repetition_penalty=1.05`, `max_new_tokens=512`. Best output
  /// quality; not bit-stable across runs.
  ///
  /// Source: <https://huggingface.co/LiquidAI/LFM2.5-VL-450M> §"Inference".
  /// If Liquid AI updates the recommended values, this constant
  /// updates with the next semver-compatible release.
  pub const fn new() -> Self;

  /// Indexing-safe greedy: `temperature=0.0` (argmax),
  /// `repetition_penalty=1.05` retained (greedy without it loops on
  /// small models). `min_p` is irrelevant under argmax (no nucleus
  /// pruning when `temperature == 0`).
  ///
  /// **Bit-stability caveat:** greedy is *necessary* for bit-equal
  /// output across runs but not *sufficient*. ORT bit-stability
  /// additionally requires:
  ///   - `ThreadOptions::with_intra_threads(1)` (parallel reductions
  ///     are non-deterministic across thread schedules);
  ///   - `ThreadOptions::with_inter_threads(1)`;
  ///   - CPU EP only — GPU EPs (CUDA, CoreML, etc.) have non-deterministic
  ///     atomics in attention kernels.
  /// `RequestOptions::deterministic()` only sets the SAMPLER side. To
  /// get end-to-end bit-stability, also pin `Options::thread` and avoid
  /// EP features. The `deterministic_run_is_idempotent` integration
  /// test (§12.2) pins this end-to-end on a CPU-only single-thread build.
  pub const fn deterministic() -> Self;

  // builder + in-place setters per field
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageBudget {
  min_image_tokens: usize,
  max_image_tokens: usize,
  max_tiles: usize,
  use_thumbnail: bool,
}

impl ImageBudget {
  /// preprocessor_config.json defaults: min=64, max=256, max_tiles=10,
  /// thumbnail on.
  ///
  /// **Field semantics — `max_image_tokens` is asymmetric across paths:**
  /// - In the **single-tile path** (`smart_resize` of the whole image),
  ///   this field bounds the produced tile's token count.
  /// - For the **thumbnail** in the multi-tile path (also `smart_resize`),
  ///   this field bounds the thumbnail's token count.
  /// - In the **multi-tile path's main tiles**, this field has NO effect.
  ///   Main tiles are uniform 512×512 (256 tokens each); the actual cap
  ///   on main-tile total is `max_tiles × 256`. A `max_image_tokens=256`
  ///   budget does NOT prevent a 4-tile image from producing 1024
  ///   main-tile tokens.
  ///
  /// The field name is misleading — it suggests a global cap but only
  /// bounds two specific paths. Flagged for v0.1 rename
  /// (`max_image_tokens_per_smart_resize` is more honest). For v0 we
  /// keep the upstream-compatible name and document the asymmetry here.
  pub const fn new() -> Self;

  /// Speed-optimized: max_image_tokens=64, max_tiles=4, thumbnail off.
  /// ~3-4× speedup, lower per-frame quality. Right for high-volume
  /// indexing where throughput matters more than per-frame fidelity.
  pub const fn fast() -> Self;

  /// Quality-optimized: explicit "all knobs to max" for clarity.
  /// Currently identical to `new()`; kept as a named preset so that
  /// future preprocessor-config changes don't silently change the
  /// "I want best quality" code path.
  pub const fn quality() -> Self;

  // builder + in-place setters per field
}
```

### 6.4 `Preprocessor` (in `preproc/mod.rs`, wasm-compatible)

```rust
pub struct Preprocessor { budget: ImageBudget }

impl Preprocessor {
  pub fn new(budget: ImageBudget) -> Self;

  /// Single-image preprocess. Each image is processed independently;
  /// callers preprocessing N images then call `preprocess` N times
  /// (or use `preprocess_batch` for the slice convenience).
  pub fn preprocess(&self, image: &DynamicImage) -> Result<PreprocessedImage>;

  /// Multi-image convenience. Returns one `PreprocessedImage` per
  /// input. Each retains its own `num_tiles` / `num_image_tokens`.
  /// `Engine::generate` / `Engine::run` use this internally and
  /// concatenate the per-image `pixel_values` along the tile axis
  /// before the single `vision_encoder.run` call.
  pub fn preprocess_batch(
    &self,
    images: &[DynamicImage],
  ) -> Result<Vec<PreprocessedImage>>;

  /// Convenience for callers who haven't decoded the image themselves.
  /// **Honors EXIF orientation** before patching — phone-camera JPEGs
  /// commonly store sensor-grid pixels with a `Rotate90CW` tag that
  /// the displayed-correct viewer applies on the way out. Without this,
  /// the VLM would receive the stored grid (sideways) and emit a
  /// confidently wrong description with no failure indicator. siglip2's
  /// `decode_with_orientation` is the reference implementation; we mirror
  /// the pattern. PNG / formats without orientation metadata fall back
  /// to `Orientation::NoTransforms`.
  ///
  /// Internally calls `lfm::decode_with_orientation(path)` to get an
  /// orientation-corrected `DynamicImage`, then routes through
  /// `preprocess(&DynamicImage)`.
  ///
  /// Gated on BOTH `feature = "decoders"` AND `not(target_arch = "wasm32")`
  /// because wasm32-unknown-unknown has no `std::fs`. Wasm callers
  /// decode the image in JavaScript and pass through `preprocess(&DynamicImage)`.
  #[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
  pub fn preprocess_path(&self, path: &Path) -> Result<PreprocessedImage>;
}

// ----- EXIF helpers (free functions, not Preprocessor methods) -----

/// Decode an image from a filesystem path, applying EXIF orientation
/// before returning. Use this when you need to construct a `DynamicImage`
/// for later use with `Preprocessor::preprocess(&DynamicImage)` — calling
/// `image::open(path)` directly skips orientation, leaving phone-camera
/// JPEGs sideways with no failure indicator.
///
/// **EXIF metadata is read from the decoder before image consumption.**
/// Once a `DynamicImage` exists, the orientation tag is gone — there's
/// no `apply_exif_orientation(&mut DynamicImage, &Path)` form because
/// re-opening the path after decode loses the orientation context that
/// the decoder already discarded.
///
/// Gated on `decoders` + non-wasm32. Mirrors siglip2's
/// `decode_with_orientation` exactly.
#[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
pub fn decode_with_orientation(path: &Path) -> Result<DynamicImage>;

/// In-memory variant of [`decode_with_orientation`] for callers that
/// have JPEG/PNG bytes (e.g., from a network response, a database row,
/// or a wasm32-side fetch). Reads the EXIF orientation tag from the
/// in-memory decoder before consumption.
///
/// Gated on `decoders` only — intended to be wasm32-compatible since no
/// filesystem is involved. **Validation gate: the CI matrix in §12.6
/// must include `cargo check --target wasm32-unknown-unknown
/// --no-default-features --features decoders`** to confirm the
/// `image/jpeg` + `image/png` decoders compile to wasm32-unknown-unknown
/// in the version we pin. If that check fails (jpeg-decoder or png crate
/// breaks on wasm), tighten the gate to
/// `#[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]`
/// and document that wasm callers must decode in JavaScript and pass
/// `&DynamicImage` through `Preprocessor::preprocess` directly.
#[cfg(feature = "decoders")]
pub fn decode_bytes_with_orientation(bytes: &[u8]) -> Result<DynamicImage>;

pub struct PreprocessedImage { /* private */ }

impl PreprocessedImage {
  // ----- ONNX-input tensors (Phase 0 Gate A confirmed shapes) -----
  //
  // CRITICAL: vision_encoder.onnx takes PRE-PATCHIFIED, FLATTENED input —
  // NOT image-shaped (N, 3, H, W) tensors. Each 16×16 RGB patch is
  // flattened to a 768-vector (16*16*3 = 768) and stored as one row of
  // pixel_values. This was confirmed by inspecting the actual ONNX graph;
  // see tests/fixtures/onnx_io_contract.json.
  //
  // LAYOUT INVARIANT: padding is at the PATCH level (via attention_mask),
  // not the pixel level. Different tiles within one image can have
  // different patch counts (main tiles 1024 patches each from 512×512;
  // thumbnail 768 from 512×384 etc.). The upstream Lfm2VlImageProcessor
  // packs each tile as one batch entry, padded to per-batch num_patches
  // max with pixel_attention_mask = 0 for padded patch slots.
  pub fn pixel_values(&self) -> &[f32];          // [N_batch, num_patches, 768] flattened f32
                                                  //   N_batch = per-image batch entries (typically num_tiles incl. thumbnail)
                                                  //   num_patches = per-batch-entry max (padded with zeros)
                                                  //   768 = patch_size² × channels = 16² × 3
  pub fn pixel_attention_mask(&self) -> &[i64];  // [N_batch, num_patches] flattened
                                                  //   1 = valid patch, 0 = padded
  pub fn spatial_shapes(&self) -> &[i64];        // [N_batch, 2] = (h_patches, w_patches) per batch entry
                                                  //   reflects each batch entry's UNPADDED tile shape
  pub fn batch_size(&self) -> usize;             // N_batch (number of batch entries the encoder sees)
  pub fn patches_per_entry(&self) -> usize;      // num_patches dimension (padded max)

  // ----- Tile-grid layout (needed by chat_template::expand_image_placeholders) -----
  pub fn num_tiles(&self) -> usize;              // total = main_tiles + (1 if thumbnail else 0)
  pub fn rows(&self) -> usize;                   // main tile grid rows (1 in single-tile path)
  pub fn cols(&self) -> usize;                   // main tile grid cols (1 in single-tile path)
  pub fn main_tile_size(&self) -> (usize, usize); // (h, w) — always (512, 512) in multi-tile, dynamic in single-tile
  pub fn thumbnail_size(&self) -> Option<(usize, usize)>; // None in single-tile / use_thumbnail=false; (h, w) in multi-tile + thumbnail

  // ----- Token-count breakdown for chat-template expansion -----
  pub fn tokens_per_main_tile(&self) -> usize;   // 256 in multi-tile (512×512), dynamic in single-tile
  pub fn thumbnail_tokens(&self) -> Option<usize>; // None when no thumbnail
  pub fn num_image_tokens(&self) -> usize;       // total: rows*cols*tokens_per_main_tile + thumbnail_tokens.unwrap_or(0)
}
```

No `ort`, no `tokenizers` deps — wasm-compatible without the `inference` feature.

**Multi-image flow inside `Engine`:** for `images.len() > 1`:
1. `preprocess_batch(images)` returns `Vec<PreprocessedImage>`, one per source image.
2. `Engine` calls `vision_encoder.run` **once per source image** (per-image, NOT batched — see §7.5 for why batched calls silently corrupt for multi-tile-path inputs). Per-image `image_features` outputs are concatenated in source order.
3. The chat-template renders one `<image>` placeholder per source image; expansion handles each placeholder with its corresponding image's `num_image_tokens`.
4. The embed-merge step in §7 step 5 walks `<image>` token positions left-to-right, drawing from the concatenated `image_features_total` — preserving order across images.

The chat template's `parse_content` macro emits multiple `<image>` placeholders for multi-image user content. Multi-image is a first-class case, not a special path — but the per-image vision-encoder discipline (point 2) is **mandatory**, not a performance choice. Phase 0 Gate B verified that batched multi-tile calls silently corrupt outputs (`tests/fixtures/multi_image_ordering_proof.json`).

### 6.5 `SceneTask` (in `scene.rs`, lfm-specific)

```rust
pub struct SceneTask { schema: serde_json::Value, accept_empty: bool }

impl SceneTask {
  /// Construct with the default scene-analysis prompt + schema.
  ///
  /// **OCR limitation:** the LFM2.5-VL-450M model card explicitly notes
  /// the model "is not well-suited for fine-grained OCR." Running
  /// `SceneTask` against document scans, screenshots, or text-heavy
  /// frames will produce degraded `description` / `tags` output with
  /// no failure indicator. For OCR-heavy pipelines, prefer a dedicated
  /// OCR model upstream and treat SceneTask as a complement, not a
  /// substitute.
  pub fn new() -> Self;

  pub const fn accept_empty(&self) -> bool;
  pub const fn with_accept_empty(self, val: bool) -> Self;
  pub const fn set_accept_empty(&mut self, val: bool) -> &mut Self;
}

impl vlm_tasks::Task for SceneTask {
  type Output = vlm_tasks::SceneAnalysis;
  fn prompt(&self) -> &str;               // identical SCENE_PROMPT to qwen
  fn schema(&self) -> &serde_json::Value; // identical schema to qwen
  fn parse(&self, raw: &str) -> Result<vlm_tasks::SceneAnalysis, vlm_tasks::ParseError>;
}
```

V0: parser is a verbatim port of qwen's machinery (`DetectionLabels`, `TagList`, indexable-content gate). Future tuning per the 450M's drift patterns lands here, not in `qwen`.

### 6.6 `Task` trait (re-exported from `vlm-tasks`)

```rust
pub trait Task: Send + Sync {
  type Output: Send;
  fn prompt(&self) -> &str;
  fn schema(&self) -> &serde_json::Value;
  fn parse(&self, raw: &str) -> Result<Self::Output, ParseError>;
}

#[derive(thiserror::Error, Debug)]
pub enum ParseError {
  #[error("invalid JSON: {0}")]
  Json(#[from] serde_json::Error),
  #[error("schema violation: required fields missing or null: {0:?}")]
  MissingFields(Vec<&'static str>),
  #[error("structured response had no usable fields")]
  NoUsableFields,
}
```

Identical shape to qwen's current `Task` — no behavior change for qwen consumers.

## 7. Data flow

The pipeline runs in seven stages. Same flow for `generate` and `run`; only the sampling step differs.

```
1. Preprocessor (wasm-compatible — no ort)
   per image: tile-grid (find_closest_aspect_ratio + smart_resize, see §8.3),
   resize, split into N_batch tiles (incl. optional thumbnail),
   normalize (px/255 → 2·px-1, RGB),
   FLATTEN each 16×16 RGB patch into a 768-vector (patch_size² × channels),
   pad each batch entry to per-image num_patches max (mark padded patches
   in attention_mask)
   → PreprocessedImage {
       pixel_values:         [N_batch, num_patches, 768] f32
                              ← pre-patchified! NOT image-shaped (N, 3, H, W)
                              ← 768 = 16² × 3
       pixel_attention_mask: [N_batch, num_patches] i64
                              1 = valid patch, 0 = padded
       spatial_shapes:       [N_batch, 2] = (h_patches, w_patches) per entry
                              ← each entry's UNPADDED tile shape
       num_image_tokens:     usize (post-downsample total)
     }
   PHASE-0 CONFIRMED: Phase 0 Gate A inspected vision_encoder.onnx
   directly (tests/fixtures/onnx_io_contract.json) and verified the
   input shape is [batch_size, num_patches, 768] — pre-patchified.
   The padding is at the patch level via pixel_attention_mask, NOT at
   the pixel level via image_max_size. The earlier (pre-Phase-0)
   pixel-padding LAYOUT INVARIANT was wrong; the actual semantics are
   patch-padding. (Earlier confusion came from the README's Python
   sample showing `pixel_values.numpy()` without an explicit shape;
   the upstream Lfm2VlImageProcessor flattens patches before this point.)

2. Build prompt with chat template + image expansion
   apply_chat_template(messages, tools=None, add_generation_prompt=true)
   → "<|startoftext|><|im_start|>user\n<image>{prompt}<|im_end|>\n<|im_start|>assistant\n"
   Then expand the single `<image>` placeholder into the per-tile
   structure that Lfm2VlProcessor produces:
     `<|image_start|>` + (per-tile blocks of `<image>` × tokens_per_tile)
                       + `<|img_thumbnail|>` + `<image>` × thumbnail_tokens
                       + `<|image_end|>`
   Total `<image>` token count must equal num_image_tokens.

3. Tokenize → input_ids: [1, S] i64
   Verify: count of input_ids == IMAGE_TOKEN_ID(396) matches num_image_tokens
   (catches expansion bugs early; surfaces as Error::ImageTokenCountMismatch)

4. Encode
   ┌─ embed_tokens.onnx ──┐
   │ in:  input_ids [1,S] │
   │ out: inputs_embeds   │
   │      [1, S, 1024]    │
   └──────────────────────┘

   For each image i in images (SEQUENTIAL — see §7.5 below for why):
   ┌─ vision_encoder.onnx (per image) ──────────────┐
   │ in:  pixel_values        [N_batch_i, num_patches_i, 768] │
   │      pixel_attention_mask [N_batch_i, num_patches_i]     │
   │      spatial_shapes      [N_batch_i, 2]                   │
   │ out: image_features      [num_image_tokens_i, 1024]       │
   │      ← rank 2! NOT [batch, num_image_tokens, 1024]        │
   │      ← post-projector, ready-to-merge                     │
   └────────────────────────────────────────────────┘

   Concatenate image_features from each image in source-image order:
     image_features_total = concat([img0_features, img1_features, …]) along axis 0
     shape: [Σ num_image_tokens_i, 1024]

5. Embed-merge
   For each pos i where input_ids[0, pos] == 396:
     inputs_embeds[0, pos, :] = image_features_total[k++, :]
   (No new tensor; mutate inputs_embeds from step 4 in place.)

   ORDERING INVARIANT (Phase 0 G6 RESOLVED — see §7.5):
   image_features_total rows are in source-image order:
     image 0's tokens (rows 0..num_image_tokens_0)
     image 1's tokens (rows num_image_tokens_0..num_image_tokens_0+num_image_tokens_1)
     …
   Within each image's per-encoder-call output, rows are in tile-then-
   thumbnail order matching the upstream Lfm2VlImageProcessor's tile
   packing. This is enforced because Engine calls vision_encoder.run
   ONCE PER IMAGE — see §7.5.

6. Decoder loop (G1 RESOLVED — position_ids omitted)
   step 0 (prefill):
     decoder.run(inputs_embeds=[1,S,1024],
                 attention_mask=[1,S],
                 ← NO position_ids! Confirmed via Phase 0 Gate A.
                   Decoder derives positions internally from past_len
                   and the current inputs_embeds shape.
                 past_conv.{i} for i in [0,1,3,4,6,7,9,11,13,15] = zeros [1, 1024, 3]
                                                                   ← SPARSE indices
                                                                     (matches layer_types
                                                                     raw indices, NOT
                                                                     compacted 0..9)
                 past_key_values.{i}.{key,value} for i in [2,5,8,10,12,14] =
                                                            zeros [1, 8, 0, 64]
                                                            ← LENGTH-0 (grows)
                 )
     → logits[1,S,65536], present_conv.* + present.*.{key,value}
     sample(logits[0, S-1, :]) → next_token
     KvCache.advance(present_*) → next-step cache inputs

     IMPORTANT: the two cache species initialize differently.
     - past_conv.* are STATIC SHAPE [1, hidden=1024, conv_L_cache=3], zero-filled.
       The conv-state ring buffer always holds the last 3 tokens; before step 0
       there are no tokens, so the buffer is all zeros — but it's still shape [1, 1024, 3],
       NOT shape [1, 1024, 0]. Verified against the upstream Lfm2ShortConv.slow_forward,
       the transformers.js initialization code, AND tests/fixtures/onnx_io_contract.json.
     - past_key_values.{i}.{key,value} are DYNAMIC SHAPE [1, num_kv_heads=8, past_len, head_dim=64]
       with past_len=0 at step 0 (zero-length empty tensor — standard transformer KV cache).

   step k > 0 (incremental decode):
     embed_tokens.run(input_ids=[[next_token]]) → next_embed [1,1,1024]
     decoder.run(inputs_embeds=next_embed,
                 attention_mask=[1, S+k],
                 ← still no position_ids
                 past_conv.* = KvCache.conv,
                 past_key_values.* = KvCache.attn)
     → logits[1,1,65536], present_*
     sample(logits[0, 0, :]) → next_token

   Stop when (checked at TOP of each loop iteration to short-circuit
   one wasted decoder.run):
     next_token == EOS_TOKEN_ID(7)
     OR k == max_new_tokens
     OR (run only) constraint.is_stopped()  ← schema-complete

   Edge cases:
     - max_new_tokens == 0 is rejected at validate_request_options time
       with Error::InvalidRequest("max_new_tokens must be > 0") — never
       reaches the loop. (See §13.2 #19 for the full validation rules.)
     - llguidance dead-end (compute_mask all-zero) returns
       Error::LlGuidanceDeadEnd, NOT a stop. The caller sees a schema
       bug surfaced as an actionable error rather than a silent stop.
     - If is_stopped() and EOS would fire on the same step, we treat
       schema-complete as the winning stop reason (run() returns parsed
       T::Output rather than waiting one more decoder.run).

7. Detokenize + parse
   tokens → text (skip_special_tokens=true)
   generate: return text
   run:      task.parse(&text) → T::Output
```

### 7.5 Multi-image vision encoder contract (Phase 0 G6 RESOLVED — FAILED)

**Phase 0 Gate B verified that batched multi-image inference DOES NOT preserve source-image order on this ONNX export.** See `tests/fixtures/multi_image_ordering_proof.json`:

| Case | Inputs | Result | max abs diff |
|---|---|---|---|
| 1 | 256² + 256² (both single-tile path) | **PASSED** | 0.000 |
| 2 | 1024² + 1024² (both multi-tile path) | **FAILED** | 286.875 |
| 3 | 256² + 1024² (mixed) | **FAILED** | 473.250 (image 0 corrupted) |

The single-tile-path-only case works in batch, but anything routing through the multi-tile path corrupts when batched. Root cause unknown — could be batch-dependent layer norm, tile-index reordering across batch entries, or cross-image attention bleed in the encoder. The fix is at our level, not the encoder's.

**Contract for the Rust implementation:** `Engine::generate` / `Engine::run` MUST call `vision_encoder.run` **once per source image** when `images.len() > 1`. The Engine concatenates the per-image `image_features` outputs in source order before embed-merge.

**Performance impact:** N-image input pays N × vision_encoder latency instead of 1 × batched. Vision encoder is ~50ms per call vs 5–30 s for the decoder loop, so the absolute cost is small. Multi-image is also uncommon (1–3 keyframes is typical for scene analysis).

**Future-proof:** if a future ONNX re-export fixes the batched case, Engine can switch to a single batched call. The fixture-freshness contract (§13.4 #18) catches this — Gate B's `all_passed` will flip true. Until then, per-image is the safe default.

## 8. Implementation details

### 8.1 KV cache management (`runtime/decoder.rs::KvCache`)

LFM2 is a hybrid model with two cache species **that initialize differently** — `KvCache::new_empty` MUST dispatch on prefix.

**Phase 0 Gate A (G2 + G3) RESOLVED:** the conv-cache uses `past_conv.{i}` (DOT, not underscore), and **layer indices are SPARSE** matching `layer_types` raw indices (not compacted 0..9). Confirmed via `tests/fixtures/onnx_io_contract.json`.

| Layer kind | Count | Indices | Cache tensors per layer | Shape | Step-0 init |
|---|---|---|---|---|---|
| conv | 10 | `0, 1, 3, 4, 6, 7, 9, 11, 13, 15` | 1 (`past_conv.{i}`) | `[1, 1024, conv_L_cache=3]` | **fixed-shape zero-fill** |
| full_attention | 6 | `2, 5, 8, 10, 12, 14` | 2 (`past_key_values.{i}.key/value`) | `[1, num_kv_heads=8, past_len, head_dim=64]` | **length-0 (`past_len=0`)** |

The conv-state ring buffer always holds the last 3 tokens; before step 0 there are no tokens, so the buffer is all zeros — but it's still shape `[1, 1024, 3]`, not `[1, 1024, 0]`. KV cache, by contrast, is the standard `past_len=0` empty tensor that grows by 1 each step. This asymmetry is verified against `Lfm2ShortConv.slow_forward` (Python reference) and the transformers.js init code.

The exact upstream layer-type literals are `"conv"` and `"full_attention"` (not `"attn"` — the spec table abbreviates for readability; code dispatches on the literal strings).

```rust
pub(crate) struct KvCache {
  // Keys are SmolStr because the names are discovered at runtime by walking
  // ort's session.inputs() / session.outputs() — those return borrowed &str
  // tied to the session's lifetime, not 'static. SmolStr inlines names ≤23
  // bytes (covers "past_conv.0" / "past_key_values.5.key" / "present.5.value"
  // and friends) so there's no per-step heap traffic on the hot path. We
  // pay one SmolStr::from(s) at session-build time per cache tensor; after
  // that the keys never reallocate.
  conv: HashMap<SmolStr, ort::value::Tensor<f32>>,  // keys: past_conv.{i}
  attn: HashMap<SmolStr, ort::value::Tensor<f32>>,  // keys: past_key_values.{i}.{key,value}
  past_len: usize,                                   // grows with each decoded token
  // present_X → past_X name map, also discovered at session-build:
  // present_conv.{i} → past_conv.{i}, present.{i}.{key,value} → past_key_values.{i}.{key,value}
  // Used by `advance()` to swap session outputs into next-step inputs.
  present_to_past: HashMap<SmolStr, SmolStr>,
}

impl KvCache {
  /// Walks decoder.inputs(); for each cache input, dispatches on name prefix:
  /// - "past_conv.*"        → zero-fill at the static shape declared in the input metadata
  ///                          (typically [1, 1024, 3])
  /// - "past_key_values.*.*" → zero-fill at the input's declared shape with past_len=0
  ///                          (typically [1, 8, 0, 64])
  /// Anything else → Error::DecoderCacheMismatch (caller fed a non-LFM2 decoder).
  /// Also walks decoder.outputs() to build the present_to_past name map.
  fn new_empty(decoder: &Session) -> Result<Self>;

  fn extend_inputs<'a>(&'a self, inputs: &mut Vec<(&'a str, TensorRef<'a, f32>)>);
  fn advance(&mut self, present_outputs: &SessionOutputs); // present_X → past_X swap via present_to_past
}
```

**Cache-tensor names are discovered at session-build time, not hard-coded.** We walk `decoder.inputs()`, group by name prefix (`past_conv.` vs `past_key_values.`), and build a `present_X → past_X` name map. Robust to minor naming variations across ONNX export versions. (Verification gate G2 in §13.4 pins the exact dot-vs-underscore convention.)

Conv-state caches are full-replace each step (the ring buffer of last 3 tokens lives inside the ONNX). Attn caches grow by 1 each step. `past_len` tracked for `position_ids` and `attention_mask` building.

### 8.2 Sampling (`runtime/sampler.rs`)

```rust
pub(crate) trait Sampler {
  /// `logits` is the per-step logit buffer (vocab_size = 65536). Mutated
  /// in place — the buffer is owned by the Engine and reused across
  /// steps to avoid per-step allocation (see §13.2 #6).
  ///
  /// Implementations may rewrite `logits` arbitrarily (mask, repetition
  /// penalty, etc.); the buffer is per-step scratch and is not persisted
  /// across calls. The Engine refills it from the next decoder run
  /// before calling `sample` again, so any state implementations need
  /// across steps must be stored in `self`, not in `logits`.
  fn sample(&mut self, logits: &mut [f32]) -> Result<u32>;
  fn is_complete(&self) -> bool;
}

pub(crate) struct FreeSampler { request: RequestOptions, generated: Vec<u32> }
pub(crate) struct ConstrainedSampler {
  /// llguidance's per-session state machine. Driven by `compute_mask`
  /// (returns the next-token bitmask) and `commit_token` (advances state
  /// after we've sampled). Constructed via `ParserFactory::create_parser`
  /// → `Constraint::new(parser)`. The `ParserFactory` itself is cached
  /// at Engine construction (see §8.6) so per-`run` startup is cheap.
  constraint: llguidance::Constraint,
  request: RequestOptions,
  generated: Vec<u32>,
}

impl ConstrainedSampler {
  fn new(
    factory: &llguidance::ParserFactory,
    schema: &serde_json::Value,
    request: RequestOptions,
  ) -> Result<Self>;
}

impl Sampler for ConstrainedSampler {
  fn sample(&mut self, logits: &mut [f32]) -> Result<u32> {
    let mask = self.constraint.compute_mask().map_err(Error::llguidance)?;
    if mask_is_all_zero(&mask) {
      return Err(Error::LlGuidanceDeadEnd {
        step: self.generated.len(),
        state: SmolStr::from(self.constraint.debug_state()),
      });
    }
    apply_mask_in_place(logits, &mask); // -inf for invalid tokens
    apply_repetition_penalty(logits, &self.generated, self.request.repetition_penalty);
    let token = if self.request.temperature == 0.0 {
      argmax(logits)
    } else {
      sample_min_p(logits, self.request.temperature, self.request.min_p)
    };
    self.constraint.commit_token(token).map_err(Error::llguidance)?;
    self.generated.push(token);
    Ok(token)
  }

  /// True once `commit_token` has signaled the schema is satisfied
  /// (llguidance's stop-result mechanism). The decode loop checks this
  /// alongside EOS and the `max_new_tokens` cap.
  fn is_complete(&self) -> bool { self.constraint.is_stopped() }
}

impl Sampler for FreeSampler {
  fn sample(&mut self, logits: &mut [f32]) -> Result<u32> {
    apply_repetition_penalty(logits, &self.generated, self.request.repetition_penalty);
    let token = if self.request.temperature == 0.0 {
      argmax(logits)
    } else {
      sample_min_p(logits, self.request.temperature, self.request.min_p)
    };
    self.generated.push(token);
    Ok(token)
  }
  fn is_complete(&self) -> bool { false }   // EOS / max_new_tokens decide for free-form
}
```

The decode loop in `generate.rs` is parameterised over `dyn Sampler` so the loop body is identical for `generate` and `run`.

**API note:** `Constraint` is the per-call state machine; `Matcher` is llguidance's server-side variant for batched generation across many requests (not relevant for single-instance sync use). We use `Constraint`.

### 8.3 Tile-grid algorithm (`preproc/tile_grid.rs`)

Direct port of upstream `image_processing_lfm2_vl.py`. The two paths are NOT symmetric:

- **Multi-tile path: tiles are uniform 512×512.** The image is resized to `(cols × 512, rows × 512)` and split into `rows × cols` uniform 512×512 tiles. Each tile produces a fixed `(512/16/2)² = 256` tokens. The optional thumbnail is dynamically sized via `smart_resize`.
- **Single-tile path: dynamic size.** The whole image goes through `smart_resize` (which constrains pixels to fit the token budget while preserving aspect ratio + ensuring dims divisible by `encoder_patch_size * downsample_factor = 32`).

The two upstream helper functions:

- **`find_closest_aspect_ratio(src_aspect, min_tiles, max_tiles, src_area)`** — enumerates (rows, cols) pairs with `rows*cols ∈ [min_tiles, max_tiles]`, picks the ratio closest to `src_aspect` (ties broken by area match).
- **`smart_resize(src_w, src_h, min_image_tokens, max_image_tokens)`** — ensures dimensions divisible by 32; constrains pixel count between `min_image_tokens * 32²` and `max_image_tokens * 32²`; preserves aspect ratio.

Path selection by `max_pixels_tolerance`: images whose pixel count exceeds `max_pixels_tolerance * max_image_tokens * 32²` go through the multi-tile path; otherwise single-tile.

```rust
fn pick_tile_grid(src_w: u32, src_h: u32, budget: &ImageBudget) -> Result<TileGrid> {
  let area = src_w as u64 * src_h as u64;
  let pixel_cap = budget.max_image_tokens as u64 * 32 * 32;
  if area as f32 > budget.max_pixels_tolerance() * pixel_cap as f32 {
    // ===== Multi-tile path =====
    // 1. Pick aspect-ratio-best (rows, cols)
    let (rows, cols) = find_closest_aspect_ratio(
      src_w as f32 / src_h as f32, budget.min_tiles, budget.max_tiles, area,
    );
    // 2. Tile dimensions are FIXED at 512×512 (per upstream; not smart_resize'd)
    let tile_w = 512;
    let tile_h = 512;
    // 3. Optional thumbnail is DYNAMICALLY SIZED via smart_resize over
    //    the source dims (not the tile dims) — gives a downscaled
    //    representation of the whole image
    let thumbnail = if budget.use_thumbnail {
      Some(smart_resize(src_w, src_h, budget.min_image_tokens, budget.max_image_tokens))
    } else {
      None
    };
    Ok(TileGrid { rows, cols, tile_w, tile_h, thumbnail })
  } else {
    // ===== Single-tile path =====
    // No splitting; the whole image is smart_resize'd to fit the budget
    let (tile_w, tile_h) = smart_resize(
      src_w, src_h, budget.min_image_tokens, budget.max_image_tokens,
    );
    // Single-tile path doesn't use a thumbnail (the single tile IS the whole image)
    Ok(TileGrid { rows: 1, cols: 1, tile_w, tile_h, thumbnail: None })
  }
}
```

**Token counts:**

- **Multi-tile**: `tokens_per_main_tile = 256` (always, since 512×512 / 16 / 2 = 16×16). Total main-tile tokens = `rows * cols * 256`. Thumbnail (if enabled) adds `(thumb_h/32) * (thumb_w/32)` tokens on top. **`num_image_tokens` = main + thumbnail.**
- **Single-tile**: `num_image_tokens = (tile_h/32) * (tile_w/32)`.

So in the multi-tile path, the main tiles ARE always 512×512 (only the thumbnail is dynamically sized). In the single-tile path, the tile is dynamically sized and there's no thumbnail. Earlier wording in this spec ("Tiles are NOT always 512×512") was wrong for the multi-tile case — the correct statement is that the **single-tile path produces dynamic dimensions; the multi-tile path produces 512×512 main tiles + a dynamically-sized thumbnail.**

Verification: capture 20+ `(src_w, src_h, ImageBudget)` → `(path, rows, cols, tile_h, tile_w, thumbnail_h, thumbnail_w, num_image_tokens)` cases from the upstream Python processor. Pin as `tests/fixtures/tile_grid_cases.json`. Test reads cases and asserts our port matches each field.

### 8.4 `<image>` expansion (`chat_template.rs`)

`apply_chat_template` returns `"<image>"` as a single placeholder per the Jinja template (one placeholder per source image — for multi-image, the chat template emits N placeholders). Expansion runs **before** tokenization, on the chat-formatted string. We follow upstream `Lfm2VlProcessor.expand_text_with_placeholders` + `_build_image_tokens`:

**Multi-tile case** (rows > 1 or cols > 1, default for non-trivial images):

```
"<image>"  →  "<|image_start|>"
              + for each (row, col):
                  "<|img_row_{row+1}_col_{col+1}|>"   ← per-tile position token
                  + "<image>" × tokens_per_tile
              + (if use_thumbnail:
                  "<|img_thumbnail|>" + "<image>" × thumbnail_tokens)
              + "<|image_end|>"
```

**Single-tile case** (the `smart_resize` path in §8.3):

```
"<image>"  →  "<|image_start|>"
              + "<image>" × tokens_per_image
              + "<|image_end|>"
```

The per-tile position tokens (`<|img_row_R_col_C|>`) come from the tokenizer's reserved-tokens block (the tokenizer.json has 300+ reserved tokens, including a grid of these). At Engine construction we verify the tokenizer contains `<|img_row_1_col_1|>` (and a representative high-row/high-col entry) — surfaces missing-token mismatches at load time, not at first request.

**API split:** `chat_template::apply_chat_template(...)` produces the raw chat-formatted string with `<image>` placeholders. A separate `chat_template::expand_image_placeholders(prompt: &str, images: &[PreprocessedImage]) -> Result<String>` walks the placeholders and substitutes the per-image expansion using the per-image grid layout from `PreprocessedImage::rows() / cols() / tokens_per_main_tile() / thumbnail_tokens()`. Returns `Error::ImageTokenCountMismatch` if the number of `<image>` placeholders in `prompt` doesn't match `images.len()`. Engine calls them in sequence inside `generate.rs`.

Pinned via `tests/fixtures/image_expansion_cases.json` — 5+ `(image_size, ImageBudget) → expected expanded prompt` cases captured from upstream Python.

### 8.5 ONNX session validation (`runtime/session.rs`)

Strict at `Engine::from_files` time. Catches mismatched ONNX exports at load, not at first decode call. Same `check_outlet` pattern siglip2 uses.

```rust
fn validate_vision_session(s: &Session) -> Result<()> {
  // Phase 0 Gate A confirmed pixel_values is PRE-PATCHIFIED: shape
  // [batch_size, num_patches, 768] where 768 = 16² × 3. NOT image-shaped.
  check_outlet(s.inputs(),  "pixel_values",         f32, &[-1, -1, 768])?;
  check_outlet(s.inputs(),  "pixel_attention_mask", i64, &[-1, -1])?;
  check_outlet(s.inputs(),  "spatial_shapes",       i64, &[-1, 2])?;
  // Output: rank 2 [num_image_tokens, 1024]. NOT rank 3.
  // Output tensor name (G4 RESOLVED): "image_features".
  check_outlet(s.outputs(), "image_features",       f32, &[-1, 1024])?;
  Ok(())
}

fn validate_embed_session(s: &Session) -> Result<()> {
  check_outlet(s.inputs(),  "input_ids",     i64, &[-1, -1])?;
  // Output tensor name confirmed by Gate A: "inputs_embeds".
  check_outlet(s.outputs(), "inputs_embeds", f32, &[-1, -1, 1024])?;
  Ok(())
}

fn validate_decoder_session(s: &Session) -> Result<()> {
  check_outlet(s.inputs(),  "inputs_embeds",  f32, &[-1, -1, 1024])?;
  check_outlet(s.inputs(),  "attention_mask", i64, &[-1, -1])?;
  // Phase 0 Gate A (G1 RESOLVED): decoder has NO position_ids input.
  // Model derives positions internally from past_len + inputs_embeds.
  // We do NOT check for position_ids — it should not be present.

  let cache_inputs = collect_cache_inputs(s.inputs())?;
  // Phase 0 G2/G3 RESOLVED: 10 conv tensors at sparse indices
  // [0, 1, 3, 4, 6, 7, 9, 11, 13, 15], 12 K/V tensors at sparse indices
  // [2, 5, 8, 10, 12, 14] × {key, value}.
  if cache_inputs.conv.len() != 10 || cache_inputs.attn.len() != 12 {
    return Err(Error::DecoderCacheMismatch { /* ... */ });
  }
  // Sparse-index check: ensure conv layer indices are exactly the 10
  // expected and attn indices are exactly the 6 expected. A different
  // sparse pattern (e.g. a model with different layer_types) would
  // fail this check at session-build, not at first decode call.
  check_sparse_indices(&cache_inputs.conv, &[0, 1, 3, 4, 6, 7, 9, 11, 13, 15])?;
  check_sparse_indices(&cache_inputs.attn, &[2, 5, 8, 10, 12, 14])?;

  // outputs: logits [batch, seq, vocab=65536] + matching present_*
  check_outlet(s.outputs(), "logits", f32, &[-1, -1, 65536])?;
  Ok(())
}
```

`-1 = required-dynamic` (rejects exports that bake static dims); other values mean exact match (or `-1` ok). Same semantics as siglip2.

**Outlet names pinned by Phase 0 Gate A** (`tests/fixtures/onnx_io_contract.json`):

| Session | Inputs | Outputs |
|---|---|---|
| vision_encoder | `pixel_values`, `pixel_attention_mask`, `spatial_shapes` | `image_features` |
| embed_tokens | `input_ids` | `inputs_embeds` |
| decoder_model_merged | `inputs_embeds`, `attention_mask`, `past_conv.{0,1,3,4,6,7,9,11,13,15}`, `past_key_values.{2,5,8,10,12,14}.{key,value}` | `logits`, `present_conv.{same}`, `present.{same}.{key,value}` |

(Decoder cache outputs use `present_conv.{i}` and `present.{i}.{key,value}` — note the `present.{i}` form for attn, NOT `present_kv.{i}`. The `present_X → past_X` mapping in `KvCache::advance` walks: `present_conv` → `past_conv`, `present.` → `past_key_values.`.)

### 8.6 llguidance integration

- **`ParserFactory` is cached lazily on the first `run` call.** It "compiles grammars and holds shared tokenizer state" (per llguidance docs). Building it per-`run` would re-compile shared infrastructure every call; building it eagerly at `Engine` construction would waste work for engines that only ever call `generate`. So Engine holds an `Option<ParserFactory>` initialized on first `run`. Subsequent `run` calls reuse the cached factory. **Build cost is 0.1–2 s** for typical schemas (SceneTask's flat object fits the low end). The first `run` call therefore has a one-time grammar-compile tax on top of the usual generation cost; subsequent calls don't. The `info`-level tracing span on `Engine::run` records this so debugging "why was the first call slower" is observable.
- **Per `run` call:** `factory.create_parser(schema)` → wrap in `Constraint::new(parser)` → drive via `compute_mask` / `commit_token`.
- **Each decode step:** `constraint.compute_mask()` → bitmask → AND with logits (set invalid tokens to `-inf`) → sample → `constraint.commit_token(token)` to advance the state machine.
- **Stop check:** `constraint.is_stopped()` checked alongside EOS and `max_new_tokens` cap.
- **`is_stopped()` semantics depend on grammar shape.** llguidance returns `true` when the grammar reaches a `stop_at` token OR the parser hits an explicit stopping rule. JSON schemas compiled via llguidance's `lark`/`json_schema` modes typically end at the closing `}` of the top-level object — but **only if the schema is authored such that the closing brace is structurally unambiguous as the grammar's accept point.** If a schema permits trailing whitespace or has nested optional regions, `is_stopped()` may not fire at the natural end-of-JSON; the decode loop would then run until `max_new_tokens` or EOS. The implementation must verify that `SceneTask::schema()` produces a grammar whose `is_stopped()` fires deterministically at the closing brace; pin via integration test.
- **Empty-mask case** (`mask_is_all_zero`) → `Error::LlGuidanceDeadEnd { step, state }` so schema bugs surface as actionable errors, not silent hangs.

**Version + wasm:** llguidance is at **1.7.5** (April 2026). Pin `llguidance = "1.7"` in the `inference` feature. **It DOES support wasm** via the `wasm` feature flag (uses `instant`). Our v0 doesn't enable wasm-side inference because `ort` and `tokenizers` are still native-only; if a future ort/tokenizers wasm story emerges, llguidance won't be the blocker.

## 9. Error taxonomy

Single `Error` enum in `error.rs` (matches siglip2/egemma; qwen's `LoadError + Error` split is a relic of mistralrs's API that we don't have here).

**Style rules:**
1. **Wrap, don't stringify.** External errors (`tokenizers`, `llguidance`) are boxed as `Box<dyn std::error::Error + Send + Sync>` to preserve source chains. Avoid `String` on the hot error path.
2. **`SmolStr` for runtime-built short strings** (matcher state, etc.) — inline allocation for <23 bytes.
3. **`&'static str` for fixed literals** (outlet names, `InvalidRequest` reason codes) — zero cost.
4. **`#[error(transparent)]`** on already-self-describing wrapped errors.
5. **Named constructors when `From` would conflict** (`Error::tokenizer(e)` / `Error::llguidance(e)` since both wrap `Box<dyn Error>`).

```rust
pub type Result<T> = std::result::Result<T, Error>;

/// `#[non_exhaustive]` so adding new variants in a future minor release
/// (notably the streaming + tool-calling variants planned in §10) is
/// not a SemVer break — downstream `match` arms must include a
/// wildcard. Mirrors siglip2/egemma. Without this, every v0.1+ variant
/// addition would force a major bump.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
  // Loading
  #[error("file not found: {0}")]
  NotFound(PathBuf),
  #[error(transparent)]
  Io(#[from] std::io::Error),
  #[error(transparent)]
  Ort(#[from] ort::Error),
  #[error(transparent)]
  Tokenizer(Box<dyn std::error::Error + Send + Sync>),

  // Session validation
  #[error("session contract mismatch on {input}: expected {expected}, got {got:?}")]
  SessionContractMismatch { input: &'static str, expected: &'static str, got: ort::value::TensorElementType },
  #[error("session shape mismatch on {input}: expected {expected}, got {got:?}")]
  SessionShapeMismatch { input: &'static str, expected: &'static str, got: Vec<i64> },
  #[error("decoder cache mismatch: expected {expected_conv} conv + {expected_attn} attn, got {got_conv} conv + {got_attn} attn")]
  DecoderCacheMismatch { expected_conv: usize, expected_attn: usize, got_conv: usize, got_attn: usize },

  // Preprocessing
  #[cfg(feature = "decoders")]
  #[error(transparent)]
  ImageDecode(#[from] image::ImageError),
  #[error("image {w}x{h} too small for ImageBudget (need at least {min_w}x{min_h})")]
  ImageTooSmall { w: u32, h: u32, min_w: u32, min_h: u32 },
  #[error("no valid tile grid for image {w}x{h} under budget {budget:?}")]
  TileGridImpossible { w: u32, h: u32, budget: ImageBudget },

  // Tokenization / template
  #[error("expected {expected} <image> tokens in expanded prompt, got {got}")]
  ImageTokenCountMismatch { expected: usize, got: usize },

  // Generation
  #[error(transparent)]
  LlGuidance(Box<dyn std::error::Error + Send + Sync>),
  #[error("llguidance produced empty mask at step {step}: {state}")]
  LlGuidanceDeadEnd { step: usize, state: SmolStr },
  #[error("hit max_new_tokens={max} (schema_complete={schema_complete})")]
  MaxTokensExceeded { max: usize, schema_complete: bool },
  #[error("detokenize produced invalid UTF-8")]
  InvalidUtf8,
  #[error("generation produced empty output")]
  Empty,

  // Configuration
  #[error("invalid RequestOptions: {0}")]
  InvalidRequest(&'static str),
  #[error("invalid ImageBudget: {0}")]
  InvalidBudget(&'static str),

  // Task parse
  #[error(transparent)]
  Parse(#[from] vlm_tasks::ParseError),
}

impl Error {
  pub(crate) fn tokenizer<E: Into<Box<dyn std::error::Error + Send + Sync>>>(e: E) -> Self {
    Self::Tokenizer(e.into())
  }
  pub(crate) fn llguidance<E: Into<Box<dyn std::error::Error + Send + Sync>>>(e: E) -> Self {
    Self::LlGuidance(e.into())
  }
}
```

**Differences from siglip2/egemma:** Both currently stringify tokenizer errors (`Error::Tokenizer(String)`). `lfm` starts cleaner with boxed sources from day one. No retroactive change to siglip2/egemma in this design.

## 10. Streaming & tool calling — explicit deferral

Both excluded from v0, with extension points sketched:

**Streaming:**
- Free-form `generate` returns `String` synchronously.
- Adding later: `engine.generate_stream(images, prompt) -> impl Iterator<Item = Result<TokenChunk>>` (or callback variant). The decode-loop body in `generate.rs` is already factored over a `Sampler` trait — adding a yield-after-each-token path doesn't touch `Sampler` or `KvCache`.
- `run` deliberately doesn't stream — the parser needs the full JSON.

**Tool calling:**
- The model has `<|tool_call_start|>` / `<|tool_call_end|>` tokens; the chat template has a `tools` slot.
- Adding later: `engine.generate_with_tools(images, prompt, tools: &[Tool]) -> Result<ToolCallResponse>` where `ToolCallResponse = enum { Text(String), Calls(Vec<ToolCall>) }`. The Rust chat-template port already handles the `tools` slot internally — we just don't expose it in v0.
- Adding tool calling does NOT require touching `runtime/*` or the `Sampler` trait.
- **Note:** Liquid AI documents tool-calling as **text-only** (BFCLv4 21.08). Combining `images` with `tools` is outside their tested envelope — when v0.1 adds the API, the doc-comment should warn callers that image+tool combined inputs are unsupported by Liquid AI's evaluation, even though the chat template structurally permits it.

## 11. Bundling strategy

| Asset | Size | Strategy |
|---|---|---|
| Preprocessing constants (mean/std/sizes) | 0 (Rust code) | Always baked in as `const` |
| Special-token strings + IDs | 0 (Rust code) | Always baked in as `const` |
| Sampling defaults | 0 (Rust code) | Always baked in (`RequestOptions::new()` / `deterministic()`) |
| `apply_chat_template` Rust function | 250–400 LoC (revised; see §13.3 #12) | Always baked in. **Reconsider `minijinja` at impl time** — see §13.3 #12 for the trade-off |
| `chat_template.jinja` source | 3.8 KB | Always shipped + exposed as `BUNDLED_CHAT_TEMPLATE_JINJA` (reference / parity-test fixture) |
| `preprocessor_config.json` | 0.7 KB | Shipped in `models/` as a build-time fixture only. A unit test parses it and asserts `ImageBudget::new()` matches each field — catches drift between Rust constants and the upstream config. **Not exposed publicly** (no consumer benefit). |
| `tokenizer.json` | 4.5 MB | Behind `feature = "bundled"` (default ON) — `BUNDLED_TOKENIZER` + `Engine::bundled` constructor |
| ONNX weights (vision/embed/decoder) | hundreds of MB | Never bundled — caller passes paths |

**Files NOT shipped** in `models/`: `config.json`, `generation_config.json`, `processor_config.json`, `tokenizer_config.json`. Every value we need from these is baked into Rust constants (BOS/EOS/PAD ids, image_token_id, special-token strings, model dimensions). Shipping them as 13 KB of LFM-licensed payload that no caller and no test reads is dead weight; if the licensing context ever changes, every byte we don't ship is one fewer compliance question.

Compressed `.crate` size estimate: ~1.8–2.2 MB (`tokenizer.json` 4.5 MB compresses to ~1.7–2.0 MB on its own; rest is small). Well under the 10 MB crates.io limit either way. Will measure exactly at first publish. `--no-default-features` opts out of `inference + bundled + decoders`; `--no-default-features --features inference` keeps the engine but drops the 4.5 MB tokenizer (caller must pass a path).

## 12. Testing strategy

### 12.1 Unit tests (no model required)

| Module | Tests |
|---|---|
| `chat_template.rs` | `apply_chat_template` against 8–10 fixture inputs; bit-equal to upstream Jinja. |
| `preproc/tile_grid.rs` | 20+ cases captured from upstream Python; pin grid + `num_image_tokens`. |
| `preproc/mod.rs` | Normalization, NCHW layout, mask shape, thumbnail inclusion (synthetic 4-pixel images). |
| `runtime/sampler.rs` | Greedy = argmax; min_p filter correctness; mask zeroes correct tokens; rep-penalty math. |
| `runtime/decoder.rs` | `KvCache::new_empty(decoder)` discovers names off stubbed `Session::inputs()`; `advance()` swaps `present_X → past_X`. |
| `runtime/session.rs` | `check_outlet` rejects: missing-outlet, wrong-dtype, wrong-rank, static-batch-where-dynamic-required, wrong-static-dim. Stubs `Outlet` values. siglip2's `image_enc.rs::check_outlet_rejects_static_batch_dim` is the verbatim reference. |
| `scene.rs` | Verbatim port of qwen's parser tests. Same SCENE_PROMPT prompt-hygiene assertion. |
| `options.rs` | `RequestOptions::deterministic()` is greedy; `new()` matches model card; `ImageBudget` presets distinct; validation rejects bad inputs. |
| `error.rs` | `LlGuidanceDeadEnd::state` fits inline in SmolStr; Display matches `#[error(...)]`; named constructors round-trip through `Box<dyn Error>`. |
| auto-traits | Compile-time `req::<T>()` assertions for `Engine: Send`, `Preprocessor: Send + Sync`, `SceneTask: Send + Sync`, `Options: Send + Sync`, `RequestOptions: Send + Sync + Copy`, `ImageBudget: Send + Sync + Copy`. Catches accidental `Rc`/`!Send` regressions at compile time. (siglip2's `tests/integration.rs:48` is the reference idiom.) |

Target: **80+ unit tests**, < 2 s total, runnable in CI on every commit, no GPU/model dependency.

### 12.2 Integration tests (gated on `integration` feature)

```rust
// tests/integration.rs — requires LFM_MODEL_PATH env var
// Skips with `eprintln!("LFM_MODEL_PATH unset, skipping")` if missing.

#[test] fn smoke_generate_text_only();              // generate(&[], "what is 2+2?")
#[test] fn smoke_generate_one_image();              // generate(&[airport_01], "describe this")
#[test] fn structured_scene_task();                 // run(SceneTask, &keyframes) → SceneAnalysis
#[test] fn deterministic_run_is_idempotent();       // bit-equal SceneAnalysis across runs
#[test] fn empty_images_text_only_works();
#[test] fn max_tokens_cap_returns_max_tokens_exceeded();
#[test] fn over_constrained_schema_returns_dead_end();

#[test]
fn schema_stops_at_closing_brace_not_max_tokens() {
  // §8.6 contract: SceneTask::schema() must produce a grammar whose
  // is_stopped() fires deterministically at the closing brace of the
  // top-level JSON object — NOT trigger a max_new_tokens stop.
  let opts = RequestOptions::deterministic().with_max_new_tokens(2048);
  let result = engine.run_with(&SceneTask::new(), &fixtures, &opts)?;
  // If is_stopped fired correctly, generation completes well below the cap;
  // if it didn't (and we hit max_new_tokens instead), the parser would
  // typically also fail with mid-string truncation.
  // Pin: the call returns Ok(SceneAnalysis), proving both:
  //   1. is_stopped fired (otherwise we'd hit MaxTokensExceeded)
  //   2. the output was schema-valid JSON (otherwise parser would fail)
  assert!(!result.description().is_empty());
}
```

Run with: `LFM_MODEL_PATH=/path/to/model cargo test --features integration --test integration -- --test-threads=1`. Off by default; CI doesn't have model weights.

**`--test-threads=1` is mandatory** because the ort runtime is process-global and concurrent `Engine` constructions can race on session-builder state. Same constraint siglip2's integration tests have. Without it, intermittent failures show up as session-build errors that look like "ONNX file is corrupt" but are actually ort initialization races.

### 12.3 Fixtures (`tests/fixtures/`)

```
airport_01.jpg, airport_02.jpg, airport_03.jpg     # ported from qwen for A/B (multi-image case)
chat_template_cases.json                            # 8-10 cases for template parity (incl. multi-image)
tile_grid_cases.json                                # 20+ cases for grid algorithm parity (single + multi-tile + thumbnail variants)
image_expansion_cases.json                          # 5+ cases for <image> expansion (incl. 2-image case
                                                    #   for the embed-merge ordering invariant)
onnx_io_contract.json                               # Phase-0 capture from `python scripts/capture_onnx_io.py`
scene_payloads/                                     # parser tests (canonical, edge-cases)
```

**Multi-image fixture coverage is required**, not optional. The embed-merge ordering invariant (§7 step 5) is silently corruptible — only a 2+image fixture catches a vision_encoder that reorders by tile-index across images. The parser-only tests can't detect this; only an end-to-end `Engine::run(&task, &[img_a, img_b])` integration test against a real model can.

### 12.4 Examples (`examples/`)

```
smoke.rs            # phase-zero "does it work" — analogous to qwen/examples/smoke.rs
scene_analysis.rs   # load + run(SceneTask, keyframes) → print SceneAnalysis
qwen_compare.rs     # both engines on same fixtures, side-by-side print
preprocess_only.rs  # wasm-compat showcase: Preprocessor only, no Engine
```

### 12.5 Benches (`benches/`)

```
bench_preproc.rs        # single-image preprocessing throughput
bench_tile_grid.rs      # grid algorithm (pure CPU)
bench_chat_template.rs  # apply_chat_template
```

**No end-to-end inference bench in v0** — GPU-bound, noisy across EPs. If we want inference latency numbers, do them in a separate harness binary with proper warmup + percentile reporting (siglip2's `examples/bench_ep.rs` is the pattern).

### 12.6 CI matrix (`.github/workflows/ci.yml`)

Mirror siglip2's:

```yaml
- cargo build --all-targets                                    # default features
- cargo build --no-default-features --features inference       # no bundled tokenizer
- cargo check --target wasm32-unknown-unknown --no-default-features                      # wasm subset (no inference, no decoders)
- cargo check --target wasm32-unknown-unknown --no-default-features --features decoders   # wasm + decoders (validates decode_bytes_with_orientation claim)
- cargo hack check --feature-powerset --exclude-features cuda,tensorrt,directml,rocm,coreml,integration
- cargo test --lib                                             # unit tests
- cargo clippy --all-targets -- -D warnings
- cargo fmt --check
- cargo test --doc                                             # rustdoc examples
```

EP-specific features excluded from the powerset (need vendor SDKs). Integration tests skipped (no model).

**Cross-platform OS/arch matrix:** the CI workflow runs the above commands across `(ubuntu-latest x86_64, macos-latest aarch64, windows-latest x86_64)` at minimum. `ort 2.0.0-rc.12` requires the bundled ONNX Runtime binary for each (Windows: needs MSVC runtime; aarch64 macOS: works out of box). siglip2/egemma's CI matrix is the reference. **Linux aarch64 explicitly NOT in v0** (ort + tokenizers cross-build is fragile on that target); add to v0.1 if a downstream caller needs it.

## 13. Open questions & risks

### 13.1 Architectural pushbacks (from adversarial review — accept-or-revisit before v0.1.0 release)

1. **Is the `vlm-tasks` shared crate worth it for v0?** Cost: new crate + qwen migration. Benefit: one canonical `SceneAnalysis`. Alternative: duplicate the ~120 LoC of `SceneAnalysis` + `Task` trait into `lfm`, defer the shared crate to v0.next once we know `lfm` is sticking. **Spec keeps the shared-crate approach** per Q6 decision, but flagged here for revisit if `qwen` migration friction shows up.

2. **`bundled` feature default-on adds 4.5 MB unconditionally.** Most callers fetch `tokenizer.json` alongside the ONNX weights — they don't need bundling. Default-on = 4.5 MB inflated binary for nothing. siglip2 has the same anti-pattern at 32 MB; we're inheriting a bad default by reflex. **Spec keeps default-on** per the user's "we should bundle some files" decision, but worth revisiting after measuring real-world download patterns.

3. **`Sampler` trait may be over-abstracted.** Two impls (`FreeSampler`, `ConstrainedSampler`) for one decode loop. Could be one function with `Option<&mut Constraint>`. Worth revisiting if the trait costs more in indirection/dyn-dispatch than it saves in clarity.

4. **No streaming for free-form `generate` is a real UX limitation.** Many VLM use cases want token-by-token feedback. Adding `generate_with_callback(images, prompt, |chunk: &str| {...}) -> Result<String>` is non-breaking — one extra method, one branch in the decode loop. **Worth adding to v0.1** if the user wants interactive use cases.

5. **`RequestOptions::deterministic()` as engine-level default is qwen-shaped, not VLM-shaped.** `lfm::Engine::generate(images, "describe this")` returning argmax-greedy output is surprising — most users expect model-card defaults. Asymmetric per-method default (greedy for `run`, model-card for `generate`) would be more honest but is also surprising in a different way. **Status quo (greedy default) ships v0**; revisit after seeing what callers actually do.

6. **Production observability beyond tracing is v0.1+, not v0.** v0 ships `tracing` spans (§13.2 #20). v0.1 should consider: latency histograms (per-call, per-stage), error rates per `Error` variant, KV-cache hit/miss for ParserFactory, generated-token counters for cost attribution, and an `Engine::health_check() -> Result<()>` doing a tiny dummy run for readiness probes. siglip2/egemma don't have these either; the broader findit-studio observability story is its own scope.

### 13.2 Specification gaps to fill in the implementation plan

6. **Logits buffer ownership.** `Sampler::sample(&mut [f32])` mutates in place. ort's `try_extract_tensor::<f32>()` returns `(shape, &[T])` (shared). The Engine holds a pre-allocated `Vec<f32>` of `vocab_size = 65536` entries; each step copies the logits slice from ort's output into this buffer before sampling. Avoids per-step 256 KB allocation.

7. **`.onnx_data` sidecar handling.** ONNX models with external weights (`decoder_model_merged_q4.onnx_data` etc.) require the sidecar to be colocated with the `.onnx` file. ort handles this transparently. `Engine::from_files` documents this in its doc comment ("if `decoder_onnx` has an `.onnx_data` sidecar, it must live in the same directory"); `Error::Io` from a missing sidecar surfaces verbatim via `#[error(transparent)]`.

8. **System prompt is deferred but the deferral is implicit.** `generate` and `run` take `prompt: &str` only — no system prompt parameter. Tools (which carry the system slot in the chat template) are deferred. **Decision for v0:** no system prompt at all; caller prepends instructions to `prompt`. Adding later: extend `Options` with `Option<String>` for engine-level default + per-call override on `*_with` methods.

9. **`apply_chat_template` Rust signature.** Concrete API in §6.1 / §11. Two functions:
    ```rust
    pub fn apply_chat_template(
      messages: &[Message<'_>],
      tools: Option<&serde_json::Value>,
      add_generation_prompt: bool,
    ) -> String;

    pub fn expand_image_placeholders(
      prompt: &str,
      images: &[PreprocessedImage],
    ) -> Result<String>;  // errors on placeholder/image-count mismatch

    // Message enum (re-exported from chat_template module)
    pub enum Message<'a> {
      System(&'a str),
      User(&'a str),                  // raw text (image placeholders stay literal)
      UserMultimodal(&'a [Content<'a>]),
      Assistant { content: &'a str /* future: thinking, tool_calls */ },
    }
    pub enum Content<'a> { Text(&'a str), Image }
    ```
    `Engine::generate` constructs the appropriate `Message` enum from `prompt: &str` + image count internally.

10. **Pixel-shuffle layout inside `vision_encoder.onnx`.** We treat it as opaque (input: `[N, 3, H, W]` per the README; output: `[num_image_tokens, 1024]` post-projector). If the ONNX export exposes intermediate steps (e.g., a separate pixel-shuffle node needing its own input), the runtime wrapper grows. **Verify at impl time by inspecting the actual `vision_encoder.onnx` graph** with `onnx.load(...).graph.input/output`.

11. **`position_ids` for image tokens.** When `<image>` expands to N tokens, those tokens get `position_ids` `seq_offset..seq_offset+N`. LFM2-VL might use 2D image-position embeddings (M-RoPE) instead of sequential — unclear from the config. **Verify at impl time by reading `modeling_lfm2_vl.py`** (specifically how `position_ids` are constructed during the forward pass with image tokens).

19. **`RequestOptions` and `ImageBudget` validation rules** (enforced at `Preprocessor::new` / `Engine::*_with_options` time; `const fn` builders can't return `Result`):
    - `RequestOptions`: `temperature >= 0.0`, `min_p ∈ [0.0, 1.0]`, `repetition_penalty >= 1.0` (values < 1 encourage repetition — almost always a caller bug), `max_new_tokens > 0`.
    - `ImageBudget`: `min_image_tokens > 0`, `max_image_tokens >= min_image_tokens`, `min_tiles >= 1`, `max_tiles >= min_tiles`, `max_pixels_tolerance > 0.0`.
    Each rule maps to a fixed `&'static str` reason in `Error::InvalidRequest` / `Error::InvalidBudget` (no allocation, callers can match on the literal).

20. **Tracing instrumentation.** All public `Engine` methods get `#[tracing::instrument]` spans. Recommended levels:
    - `info` for `Engine::run` / `Engine::generate` / `Engine::warmup` (top-level; one span per call with timing).
    - `debug` for `preprocess`, `vision_encoder.run`, `embed_tokens.run`, `decoder_loop` (per-call breakdowns; useful when debugging latency in production).
    - `trace` for per-step `decoder.run` + sample (heavyweight; only useful when debugging a specific bad output).
    Span fields include `request_kind` (run | generate), `image_count`, `prompt_token_count`, `generated_token_count`. Without span instrumentation, ops teams can't see where 25-second blocking calls spend their time.

21. **Engine-level scratch buffer reuse for preprocessing.** Each `PreprocessedImage` holds a `Vec<f32>` of `[N_batch, num_patches, 768]` (pre-patchified per §6.4 LAYOUT INVARIANT, post-Phase-0). Typical multi-tile + thumbnail case: `N_batch=5` (4 main 512×512 tiles + 1 thumbnail), `num_patches` padded to per-image max (1024 for full 512×512 tiles), 768 = 16²×3 — total `5 × 1024 × 768 × 4 bytes ≈ 15 MB`. Per-call allocation in an indexing pipeline is meaningful waste. Mirror siglip2's `embed_pixels_scratch: Option<PreprocessedBatch>` pattern: **internally to `Engine::run` / `generate`, the Engine holds an `Option<Vec<PreprocessedImage>>` scratch field that's reused across calls**; the public `Preprocessor::preprocess_batch` signature in §6.4 stays the same (returns `Result<Vec<PreprocessedImage>>` for stand-alone callers). The scratch field is an internal Engine optimization, NOT a public API change.

22. **Engine panic-safety / mid-call failure.** Engine holds multi-step state across one `run`/`generate` call (KvCache mid-generation, ParserFactory after first run, logits scratch). If a method panics or returns `Err` mid-generation, the Engine is left in an indeterminate state — the next call might start from a partial KvCache or read stale logits. Document on the `Engine` struct: **"If a method panics or returns `Err` mid-generation, discard and reconstruct the Engine rather than retry."** Future hardening (clean-on-error) is a v0.1+ task.

23. **`Engine: Send + !Sync` rationale.** `ort::Session` is `!Sync` (per ort docs). `Engine` therefore is `Send + !Sync`. Workers wanting parallelism instantiate one `Engine` per thread, or share behind a `Mutex<Engine>`. The `req::<Engine>()` compile-time assertion in §12.1 enforces `Send`; the `Engine` struct doc-comment in §6.2 should surface this so callers know the threading model up-front. (siglip2's `text_enc.rs:23` is the reference idiom — verbatim port to `engine.rs`.)

24. **`from_ort_sessions` "wasm-compatible" claim is forward-looking.** ort 2.0.0-rc.12 doesn't actually compile to wasm32 today — the function is reachable only with the `inference` feature, which pulls native-only deps. Rephrase the doc-comment to: *"Future-proof for wasm — when `ort` gains wasm32 support, this is the wasm entry point. As of `ort 2.0.0-rc.12`, requires a native target."* Don't claim a capability we don't have today.

25. **`pub use Embedding` intent.** `Embedding` is the post-projector vision-tile embedding type, used internally by `embed-merge`. We expose it publicly so callers who want raw vision-tile embeddings (e.g., for offline similarity search across tile features) can get them via a future `Engine::embed_tiles(&[DynamicImage]) -> Vec<Embedding>` accessor. If that accessor never lands, demote to `pub(crate)` in v0.1. Spec was previously silent on intent; this is the explicit answer.

### 13.3 Risk underestimations

12. **Chat-template Rust port complexity.** Earlier "~80 LoC" estimate was optimistic. The Jinja template uses macros (`format_arg_value`, `parse_content`, `render_tool_calls`), namespace state (`namespace(...)`), generation tags (`{%- generation -%}` — a tokenizers extension, not stock Jinja), `last_assistant_index` tracking, and conditional thinking-block stripping. Honest estimate: **250–400 LoC + extensive fixture coverage**. **Consider `minijinja`** (~100 KB, well-tested) instead of hand-rolling — would handle macros / namespaces / conditionals natively. **Caveat on the `generation` tag:** it's a HuggingFace `tokenizers` extension that marks the assistant generation region (used by `assistant_masks` round-tripping). "Preprocess it out" is NOT a free rewrite — stripping the tag without preserving its semantics breaks any future code that needs to know which token positions are "generation" vs "context". A minijinja port needs a custom tag handler, not a strip-and-rewrite. Worth budgeting if we go that route.

13. **`accept_empty: false` parser default may not survive contact with a 450M model.** qwen tunes this to reject sparse output as a regression signal. LFM2.5-VL is 4× smaller; sparse output may be the common case, not a regression. Risk: every `run(SceneTask)` returns `ParseError::NoUsableFields` until the user discovers `with_accept_empty(true)`. **Tune the default after measuring real parse rates** during the integration-test phase. Not a v0 blocker but a likely v0.1 follow-up.

14. **`min/max_image_tokens` field semantics are asymmetric across paths — the field name is misleading.** With `max_image_tokens=256`, `min_tiles=2`, the multi-tile path produces `rows * cols * 256` tokens regardless (uniform 512×512 main tiles), so 2 tiles = 512 tokens, 10 tiles = 2560 tokens. **`max_image_tokens` does NOT bound the multi-tile total** — it only bounds the `smart_resize` calls used in the single-tile path and for the thumbnail. The multi-tile path's actual ceiling is `max_tiles × 256 = 2560` tokens (independent of `max_image_tokens`).

    The field name `max_image_tokens` is genuinely misleading because it suggests a global cap. **Rename target for v0.1:** `max_image_tokens_per_smart_resize` (the field's actual semantic) or split into `max_main_tokens_per_image` (multi-tile) and `max_smart_resize_tokens` (single-tile + thumbnail). For v0 we keep the upstream-compatible name (`max_image_tokens`) and document the asymmetry explicitly in `ImageBudget::max_image_tokens` doc-comment.

    Fixture coverage: include `tile_grid_cases.json` entries where (a) the multi-tile path produces token counts well above `max_image_tokens` (proving the field doesn't bound it), and (b) the single-tile path is constrained by `max_image_tokens` (proving the field DOES bound that path).

15. **`pixel_attention_mask` second dim is dynamic.** Spec §8.5 originally validated `[-1, 1024]`; corrected to `[-1, -1]` because tile sizes vary in the single-tile path and the thumbnail (per the corrected §8.3). The mask is padded to per-batch max. Watch for ONNX exports that bake static dims — they'd reject anything but full 512×512 tiles.

16. **Special-tokens table is not exhaustive.** §3.1 enumerates the named special tokens (BOS, EOS, image, tool-call). The tokenizer.json carries 300+ reserved tokens — including the 100 `<|img_row_R_col_C|>` position tokens used by §8.4's expansion, plus tokens for `<|img_split|>`, `<|tool_list_*|>`, `<|tool_response_*|>`, `<|cot_*|>` (chain-of-thought), and others not surfaced by the tokenizer_config.json's `model_specific_special_tokens`. **Implementation step: dump the full added-tokens table at first impl** (`Tokenizer::added_tokens` over `tokenizer.json`); pin a unit test that verifies all tokens we name as constants in `chat_template.rs` exist with the expected IDs. Catches `tokenizer.json` re-exports that change IDs.

17. **`SceneTask::accept_empty` predicate documentation.** `with_accept_empty(true)` bypasses the indexable-content gate. The predicate is identical to qwen's: a payload is "empty" iff NEITHER (description AND tags both populated) NOR (any of subjects/objects/actions non-empty). The `SceneTask::with_accept_empty` doc-comment should explain the predicate inline (or link to qwen's `SceneTask::with_accept_empty` doc which does), so callers don't have to reverse-engineer it from parser tests. Coupled with §13.3 #13 (the default may need tuning for the 450M).

18. **Workspace lints declaration.** `Cargo.toml` snippet in §5.3 ends with `[lints] workspace = true`. This requires a workspace `Cargo.toml` at `findit-studio/Cargo.toml` declaring the lints under `[workspace.lints]`. If no workspace Cargo.toml exists (each crate is standalone), drop `workspace = true` and inline the lints under `[lints.rust]` like qwen does. **Verify at impl time:** if `findit-studio/` is a Cargo workspace, the spec's setting works; if not, change to inline.

### 13.4 Pre-implementation verification gates (Phase 0 — must complete before Rust code is written)

**Phase 0 status: COMPLETE.** Both fixture files are checked in (`tests/fixtures/onnx_io_contract.json`, `tests/fixtures/multi_image_ordering_proof.json`) and resolve all six gates. The §7 decode loop and §8.5 session validators have been updated against the captured contract. The historical "Run before any Rust code is written" framing is preserved below for future re-captures (when LiquidAI re-exports the model).

Two scripts, two distinct cost profiles:

### Phase 0 Gate A — `scripts/capture_onnx_io.py` (~30 seconds, no model weights needed)

```bash
# Resolves G1–G5 by inspecting ONNX graph metadata only.
# LFM_HF_REVISION env var should be set to the HuggingFace revision SHA
# of the LiquidAI/LFM2.5-VL-450M-ONNX repo the .onnx files came from
# (e.g., `git -C ./model rev-parse HEAD`). It anchors the
# fixture-freshness contract — `_metadata.hf_revision` is what
# `validate_*_session` cites in the loud-failure error message when a
# future re-export changes tensor names/shapes.
python3 - <<'PY'
import onnx, json, sys, os, datetime
results = {
    "_metadata": {
        "captured_at":            datetime.datetime.utcnow().isoformat() + "Z",
        "hf_repo":                "LiquidAI/LFM2.5-VL-450M-ONNX",
        "hf_revision":            os.environ.get("LFM_HF_REVISION", "unknown"),
        "capture_script_version": "1.0",
    },
}
for fname in ["vision_encoder.onnx", "embed_tokens.onnx", "decoder_model_merged.onnx"]:
    m = onnx.load(fname, load_external_data=False)
    elem = lambda t: t.tensor_type.elem_type
    shape = lambda t: [d.dim_value if d.HasField("dim_value") else d.dim_param
                       for d in t.tensor_type.shape.dim]
    results[fname] = {
        "inputs":  [(i.name, elem(i.type), shape(i.type)) for i in m.graph.input],
        "outputs": [(o.name, elem(o.type), shape(o.type)) for o in m.graph.output],
    }
json.dump(results, sys.stdout, indent=2)
PY
> tests/fixtures/onnx_io_contract.json
```

(The Gate B output JSON gets the same `_metadata` block — same revision SHA, same captured-at timestamp. Both fixtures must agree on the revision; mismatched fixtures = stale Phase 0, run both.)

### Phase 0 Gate B — `scripts/verify_multi_image_ordering.py` (~5 minutes, model weights required)

**Run Gate A first** — Gate B uses the vision encoder output tensor name resolved by Gate A's G4 (`image_features` for the current export). Without that, Gate B's `sess.run(...)[OUT_NAME]` is guessing.

```bash
# Resolves G6 by running the actual vision encoder on TWO multi-image cases.
# Requires onnxruntime + the actual ONNX weights (not a metadata-only check).
# Pseudocode:
python3 - <<'PY'
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession("vision_encoder_fp16.onnx")
OUT_NAME = "image_features"  # actual name from Phase 0 capture; re-run Gate A if a future ONNX re-export changes this

# Tolerance: fp16 weights round to ~3 decimal places; allow up to 5e-3
# absolute difference. Tighten to 1e-4 if running against the fp32 export.
fp16_tol = 5e-3

# === Case 1: single-tile-path multi-image (256×256 each, ~64 tokens) ===
# Routes through the single-tile path (area 65536 < 524288 threshold).
# Catches encoders that reorder tiles within a batch.
red, blue = build_solid_color_images(256, 256)
embeds_red_alone  = sess.run([OUT_NAME], red_inputs)[0]
embeds_blue_alone = sess.run([OUT_NAME], blue_inputs)[0]
embeds_concat_st  = sess.run([OUT_NAME], red_then_blue_inputs)[0]
assert np.allclose(embeds_concat_st[:64],     embeds_red_alone,  atol=fp16_tol)
assert np.allclose(embeds_concat_st[64:128],  embeds_blue_alone, atol=fp16_tol)

# === Case 2: multi-tile-path multi-image (1024×1024 each, ~1280 tokens) ===
# Routes through the multi-tile path: 2×2 main tiles × 256 tokens = 1024,
# plus a thumbnail. Catches encoders that interleave tile rows from
# different images instead of preserving source-image order.
red_lg, blue_lg = build_solid_color_images(1024, 1024)
embeds_red_lg_alone  = sess.run([OUT_NAME], red_lg_inputs)[0]
embeds_blue_lg_alone = sess.run([OUT_NAME], blue_lg_inputs)[0]
embeds_concat_mt     = sess.run([OUT_NAME], red_lg_then_blue_lg_inputs)[0]
n_red = embeds_red_lg_alone.shape[0]
assert np.allclose(embeds_concat_mt[:n_red],   embeds_red_lg_alone,  atol=fp16_tol)
assert np.allclose(embeds_concat_mt[n_red:],   embeds_blue_lg_alone, atol=fp16_tol)

# === Case 3: mixed-size cross-batch padding (256² + 1024²) ===
# Cases 1 and 2 both have IDENTICAL image_max within the batch, so they
# don't exercise the cross-batch repadding code path described in
# §7 step 1 NOTE. A mixed batch — small single-tile image + large
# multi-tile image — forces the encoder to handle:
#   - different per-image tile counts (red has 1 tile; blue has 4 main + thumbnail)
#   - different image_max per source image (red ≈ 224×224; blue 512×512)
#   - cross-batch padding to (max_h, max_w) over both images
# An encoder that mishandles cross-batch repadding (e.g., bleeds attention
# from the larger image into the smaller image's padded region) would
# pass cases 1+2 and fail case 3.
red_sm  = build_solid_color_images(256, 256)
blue_lg = build_solid_color_images(1024, 1024)
embeds_red_sm_alone  = sess.run([OUT_NAME], red_sm_inputs)[0]
embeds_blue_lg_alone = sess.run([OUT_NAME], blue_lg_inputs)[0]
embeds_concat_mixed  = sess.run([OUT_NAME], red_sm_then_blue_lg_inputs)[0]
n_red = embeds_red_sm_alone.shape[0]
assert np.allclose(embeds_concat_mixed[:n_red], embeds_red_sm_alone,  atol=fp16_tol)
assert np.allclose(embeds_concat_mixed[n_red:], embeds_blue_lg_alone, atol=fp16_tol)

# If any case fails, the embed-merge in §7 step 5 silently corrupts.
PY
> tests/fixtures/multi_image_ordering_proof.json
```

**Why three cases:** the single-tile case (Case 1, 256×256) catches tile-index-batching across images. The uniform multi-tile case (Case 2, 1024×1024) catches interleaving of tile rows from different images. The mixed case (Case 3, 256+1024) catches mishandled cross-batch repadding — image_max varies per source image and the batch tensor is padded to the per-batch max. A vision encoder could pass any one or two cases and fail the third.

Both fixture files MUST exist in the repo before the implementation plan's first Rust-code task can begin. (Implementation-plan enforcement: list as `pre-flight: tests/fixtures/onnx_io_contract.json && tests/fixtures/multi_image_ordering_proof.json` before any `cargo` step.)

**Fixture freshness:** these fixtures pin a specific ONNX export. If Liquid AI re-exports with changed tensor names or shapes, our `validate_*_session` checks must fail loudly with a message that reads from the fixture's `_metadata.hf_revision` and `_metadata.captured_at`, e.g.: `"spec assumes ONNX export from {hf_repo}@{hf_revision} captured at {captured_at}; current session for {input_name} differs: expected {fixture_value}, got {current_value}"` — NOT silently fall through to a different code path. The `Error::SessionShapeMismatch` / `SessionContractMismatch` variants already give us the structure for the per-input message; the metadata fields anchor the "from when" half. (The capture script writes `_metadata.hf_revision` from the `LFM_HF_REVISION` env var; if unset, value is `"unknown"` and the validator's loud-failure surfaces that explicitly so a missed Phase-0 step is obvious.)

The two gates resolved these specific facts:

**Gate A (metadata) — RESOLVED G1–G5** (`tests/fixtures/onnx_io_contract.json`):

- **G1 RESOLVED: decoder has NO `position_ids` input.** Confirmed via Gate A — the only inputs are `inputs_embeds`, `attention_mask`, and the cache tensors. Model derives positions internally from `past_len` + `inputs_embeds` shape. §7 step 6 + §8.5 `validate_decoder_session` updated.
- **G2 RESOLVED: conv-cache uses `past_conv.{i}` (DOT, not underscore).**
- **G3 RESOLVED: cache layer indices are SPARSE per `layer_types` raw indices.** Conv layers at `[0, 1, 3, 4, 6, 7, 9, 11, 13, 15]`, attn layers at `[2, 5, 8, 10, 12, 14]`. NOT compacted 0..9. §8.1 + §8.5 updated.
- **G4 RESOLVED: vision encoder output tensor name = `image_features`.** Embed_tokens output name = `inputs_embeds`. §8.5 placeholders replaced with literals.
- **G5 RESOLVED: `attention_mask` shape = `[batch_size, total_sequence_length]`** (both dims dynamic). `position_ids` doesn't exist (per G1).

**Gate A also revealed three SURPRISES requiring spec rewrites (already applied):**
- Vision encoder INPUT `pixel_values` is `[batch_size, num_patches, 768]` — pre-patchified, NOT `[N, 3, H, W]` image-shaped! 768 = 16² × 3.
- Vision encoder OUTPUT `image_features` is rank 2 `[num_image_tokens, 1024]`, NOT rank 3.
- `pixel_attention_mask` is `[batch_size, num_patches]` (per-batch-entry padding, not pixel-level).

**Gate B (runtime) — G6 RESOLVED: FAILED** (`tests/fixtures/multi_image_ordering_proof.json`):

- **G6 FAILED: multi-image batched calls silently corrupt for any image routing through the multi-tile path.** Case 1 (256²+256², single-tile) passed with diff 0.0; Case 2 (1024²+1024², multi-tile) failed with max diff 286.875; Case 3 (256²+1024², mixed) failed with max diff 473.25. **Implementation MUST call `vision_encoder.run` once per source image** (see §7.5 for the full contract). The embed-merge invariant in §7 step 5 is preserved by the per-image-call discipline, NOT by anything the encoder guarantees.

Plus two non-blocking items that stay "during impl":

16. **Cross-engine parity test** — currently a runnable example, not a CI test (CI doesn't have either model). Once we have weights staged for CI, promote to a real test.

17. **`tests/fixtures/onnx_io_contract.json` becomes a regression pin.** A future ONNX re-export with changed names/shapes blows up this fixture before it blows up `validate_*_session` at load time.

18. **Fixture maintenance cadence (v0.1+ scope, but tracked here so it doesn't get lost).** LiquidAI may re-export `LFM2.5-VL-450M-ONNX` with bug fixes or quantization changes. The fixture-freshness loud-failure (§13.4) protects against silent breakage at runtime, but the fixture-update workflow needs an owner: (a) who runs `capture_onnx_io.py` against a new revision; (b) is there a CI step that re-captures + diffs against the pinned fixture; (c) is there a renovate-bot-style automation watching the upstream HF revision SHA?

    **v0** (manual, crate-publisher-owned):
    - When `validate_*_session` fires `SessionShapeMismatch` citing the old `_metadata.hf_revision`, the **crate publisher** (not the downstream consumer) reruns Phase 0 against the new export. Downstream consumers can't self-serve — they file an issue and wait for an upstream patch release. This maintenance burden falls on the crate maintainer, not the user.
    - Procedure: `git -C ./model fetch && git -C ./model checkout <new_revision> && git -C ./model lfs pull && LFM_HF_REVISION=$(git -C ./model rev-parse HEAD) python3 scripts/capture_onnx_io.py --onnx-dir ./model/onnx > tests/fixtures/onnx_io_contract.json && python3 scripts/verify_multi_image_ordering.py --onnx ./model/onnx/vision_encoder_fp16.onnx > tests/fixtures/multi_image_ordering_proof.json`.
    - **Post-re-capture audit protocol** (catches secondary spec drift): after committing fresh fixtures, audit these spec sections for shape/name descriptions that may have drifted from the new reality:
      - §6.4 PreprocessedImage layout + Multi-image flow paragraph
      - §7 step 1 shapes
      - §8.5 validate_*_session outlet names + shapes
      - §13.2 #21 scratch-buffer shape estimate
      - §14 cheat sheet memory-per-call line
      The post-Phase-0 review (round 9) demonstrated this drift risk: the canonical sections were updated; secondary/derived sections weren't, and contradicted each other for one round before being caught.

    **v0.1+**: consider GitHub Actions cron checking `huggingface_hub.list_repo_commits()` for new revisions, opening a PR with refreshed fixtures + a checklist for the post-re-capture audit.

## 14. Appendix — Model facts cheat sheet

| Fact | Value |
|---|---|
| Architecture | `Lfm2VlForConditionalGeneration` |
| Parameters | 450M (350M LM + 86M vision + projector) |
| LM hidden | 1024 |
| LM layers | 16 (10 conv + 6 full_attention) |
| Conv state cache | `conv_L_cache=3` per conv layer |
| Vision hidden | 768 (post-projector → 1024) |
| Vision patch size | 16 |
| Tile size (max) | 512×512. Tiles are dynamically sized smaller by `smart_resize` to fit the image-token budget; 512×512 is the upper bound, not the typical case. |
| Patches per full 512×512 tile | 256 (after 2× pixel-shuffle downsample, from 1024) — theoretical max |
| Vocab size | 65536 |
| Max position embeddings | 128000 |
| GQA | 16 attention heads, 8 KV heads, head_dim 64 (1024 hidden / 16 heads) |
| BOS token | `<|startoftext|>` (id 1) |
| EOS token | `<|im_end|>` (id 7) |
| PAD token | `<|pad|>` (id 0) |
| Image token | `<image>` (id 396) |
| Sampling | `temperature=0.1`, `min_p=0.15`, `repetition_penalty=1.05` (model-card default) |
| Image budget defaults | min=64, max=256 image tokens; min=2, max=10 tiles; thumbnail on |
| Image normalization | mean=std=0.5 → pixels in `[-1, 1]` |
| Native structured output | None (prompt engineering + tool calling only) |
| Memory at idle (recommended fp16+q4 combo) | ~770 MB resident (vision_encoder fp16 ~180 MB + embed_tokens fp16 ~128 MB + decoder q4 ~459 MB) |
| Memory peak per call | + KV cache (grows with prompt + generated tokens; ~6 KB per token at GQA 8/64) + logits scratch (~256 KB) + per-image preprocess scratch (~15 MB at 5-tile multi-tile + thumbnail; pre-patchified `[N_batch=5, num_patches≤1024, 768]`) |
| Cumulative startup (load + warmup) | 3–8 s blocking (operators must size readiness probes accordingly) |

---

## 15. Model weights license — surface to callers

**The wrapper code (`lfm` crate) is dual-licensed `MIT OR Apache-2.0`. The model weights are NOT.**

LFM2.5-VL-450M ships under the **LFM Open License v1.0** (custom license, listed as `lfm1.0` in the HuggingFace model metadata). The wrapper's permissive license does NOT extend to the weights — a downstream caller assuming "this crate is MIT, so I can use the model commercially" would be making a compliance mistake.

**License URL:** <https://www.liquid.ai/lfm-license>
**Model card license declaration:** <https://huggingface.co/LiquidAI/LFM2.5-VL-450M> (`License: lfm1.0`)

### What the boundary is

| Component | License | Source |
|---|---|---|
| `lfm` crate code (everything in `src/`, `examples/`, `benches/`, `tests/`) | MIT OR Apache-2.0 | This repository |
| `vlm-tasks` crate code | MIT OR Apache-2.0 | This repository |
| `models/tokenizer.json` (bundled tokenizer) | LFM Open License v1.0 | LiquidAI HuggingFace |
| `models/chat_template.jinja` + `models/preprocessor_config.json` (build-fixture) | LFM Open License v1.0 | LiquidAI HuggingFace |
| ONNX weight files (caller-supplied, not bundled) | LFM Open License v1.0 | LiquidAI HuggingFace |

### What we surface to callers

1. **README §License** has a dedicated subsection explicitly distinguishing wrapper-code license from weights license, with a pointer to the LFM Open License URL.
2. **`Engine::from_files`, `Engine::bundled`, `Engine::from_ort_sessions` doc-comments** include a one-line note: "The model weights this Engine wraps are governed by the LFM Open License v1.0 (https://www.liquid.ai/lfm-license), separate from this crate's MIT/Apache-2.0 dual license. Verify your use case complies with Liquid AI's terms."
3. **Crate-level doc comment** (`//!` block in `src/lib.rs`) repeats the boundary at the top.

### What we do NOT do

- We do not paraphrase the LFM license or claim to interpret its terms (e.g., revenue thresholds, commercial-use rules). The reviewer's claim of a "$10M USD annual revenue threshold" is plausible based on common Llama-style community licenses but unverified by us — callers must read the actual license text.
- We do not block compilation or runtime based on license terms; that's not the wrapper's job.
- We do not bundle the weight files (only the tokenizer + `chat_template.jinja` + the `preprocessor_config.json` build-fixture); the weights stay caller-supplied via `from_files`/`bundled` paths.

---

**Implementation plan: TBD.** This spec is the input; the next step is to invoke the `writing-plans` skill to produce a step-by-step implementation plan with verification checkpoints. **Phase 0 (run `scripts/capture_onnx_io.py`) must complete before code is written** — see §13.4.
