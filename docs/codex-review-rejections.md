# Codex Adversarial Review — Rejected Findings

This file records findings that **previous adversarial reviews have raised, that we have verified to be false positives**, and the evidence that proves it. Codex re-flags these items in subsequent rounds; instead of reverifying each time, refer to the listed evidence.

Future reviewers (codex or human): if you're about to flag one of these, FIRST consult the cited evidence and confirm the bug truly exists. If you can construct a reproducer that contradicts the evidence below, file it as a NEW finding with the reproducer attached.

---

## Trust model and review scope

**Trust model.** This crate operates inside a trusted boundary on BOTH sides:

1. **The model files are trusted.** `Engine::from_dir` / `Engine::from_onnx_dir` / `Engine::from_paths` callers obtain model assets from a known source (HuggingFace, vendored bundle, internal registry) and vouch for them the same way they vouch for any other dependency.
2. **The caller-supplied inputs are trusted.** Chat messages, image bytes, image paths, `RequestOptions`, `ImageBudget`, `Task` schemas — all of these come from the same trusted application that loaded the model. The crate is an in-process library, not a public network endpoint.

In other words: **there is no attacker in this threat model**. We are not hardening against malicious model providers, malicious image uploaders, malicious chat content, or malicious schema definitions. Defense-in-depth against any of those is out of scope.

### What's IN SCOPE

These are the things to flag — strictly **CODE** issues:

- **Algorithmic correctness against upstream Python.** smart_resize math, tile_grid selection, marker order vs `crop_image_to_patches`, image-block layout, normalization constants, BPE tokenization parity, chat template rendering parity.
- **Model input/output contracts.** ONNX session input/output shapes, dtype, axis order. Embedding splice positions. Sampler math (softmax stability, repetition penalty signs, min_p threshold). KV cache shape and lifecycle. Position IDs and attention masks if applicable.
- **Bugs reachable with valid inputs.** Panics, integer overflow, wrong indexing, wrong loop bounds, off-by-one in sequence handling — when triggered by code paths a normal correct caller can hit.
- **Concurrency hazards** in code reachable through the public API: data races, deadlocks, ordering bugs.
- **Cargo manifest correctness** — feature graph, dep version constraints, lints. These compile-time guarantees ARE code.

### Also out of scope: CI / workflow / build-infrastructure findings

Findings about `.github/workflows/*`, GitHub Actions configuration, build-system scripts, repo layout (sibling-checkout requirements, missing `.gitmodules`, sibling repo path-dep resolution at CI time, etc.), or anything that's "the CI workflow doesn't do X" are **out of scope for code review**. Those are infrastructure decisions handled separately. Do NOT flag them.

### What's OUT OF SCOPE — do NOT flag

Past rounds repeatedly flagged variations of these themes. They are settled:

1. **"What if a custom tokenizer/template/config/ONNX is tampered or drifted?"** `from_paths` is documented as the unchecked escape hatch. `from_dir` already enforces byte-equality against bundled tokenizer.json, chat_template.jinja, preprocessor_config.json, and config.json's `max_position_embeddings`. Adding more strict checks for additional fields (tokenizer_config.json chat_template field, processor_config.json fields, generation_config.json fields, model config field-by-field) is not warranted.
2. **"What if a custom ONNX file's metadata is loose enough to admit a drifted runtime shape?"** ORT session validation already pins input/output names and rank-and-relevant-dim. Exhaustive exact-shape checks at every call site for every dynamic axis is defense-in-depth against a model provider who already controls the model file.
3. **"What if the on-disk tokenizer file is replaced between Engine construction and a later call?"** Round 35 already addressed this for the `from_onnx_dir` temp tokenizer. Treating local disk as a hostile attacker is outside the trust model.
4. **"What if a from_paths caller passes a tokenizer with `special: false` on every control token?"** `from_paths` is opt-out by documentation; the denylist already seeds structural + named control tokens unconditionally.
5. **"What if the caller passes a huge text payload / millions of empty parts / millions of image inputs?"** The caller is trusted. They're allocating their own process's memory. There is no remote attacker. DoS hardening of the public API is out of scope.
6. **"What if the caller passes image bytes that decode to a huge buffer (e.g., 16-bit PNG bypassing the 4-BPP header estimate)?"** Same as above. The caller chose to feed those bytes into their own process.
7. **"What if the caller sets `repetition_penalty = f32::MAX` and exploits the sampler?"** Same trust model. The caller is configuring their own sampler. We already cap at `MAX_REPETITION_PENALTY=100.0` for ergonomic correctness, not for adversarial robustness.

If a finding is on a theme above, **do not file it**. If you are unsure whether a finding is in scope, prefer the algorithm/model-I/O lens: "would this still be a bug if my caller and my model were both perfectly cooperative?" If yes, flag it. If the finding only manifests when someone misuses the API on purpose, skip it.

---

## REJECTED-1: "Tile markers and vision features use different traversal orders" (rounds 13, 14, 22)

### Claim
`build_image_block` emits markers with `cols` outer and `rows` inner, while `flatten_to_patches` builds tiles row-major (`rows` outer, `cols` inner). For non-square multi-tile grids, marker order and feature order diverge — feature k binds to marker for a different spatial position. Token counts still match so `ImageTokenCountMismatch` doesn't fire.

### Why this is NOT a bug
The "weird" pairing IS upstream Python's actual behavior, replicated bit-for-bit. The semantic confusion comes from upstream's variable-naming inversion at `transformers/models/lfm2_vl/processing_lfm2_vl.py:161-162`:

```python
images, num_rows, num_cols = self.crop_image_to_patches(...)
# but crop_image_to_patches actually returns (..., grid_width, grid_height)
# so num_rows = grid_width and num_cols = grid_height
```

In `expand_text_with_placeholders` the iteration is then:
```python
for row in range(rows):     # rows = num_rows = grid_width
    for col in range(cols): # cols = num_cols = grid_height
        emit f"<|img_row_{row+1}_col_{col+1}|>"
```

So upstream's outer loop is over `grid_width` (cols-outer in our terms). The model was trained on this convention. Our marker emission `for outer in 0..img.cols() { for inner in 0..img.rows() }` matches upstream exactly.

The model uses `masked_scatter` (modeling_lfm2_vl.py:281) for positional splicing of the k-th feature into the k-th `<image>` token, identical to our splice loop. Whatever pairing convention upstream produces, the model was trained on it; our replication is correct.

### Evidence
- `tests/fixtures/image_expansion_cases.json::multi_tile_4x2_widescreen` — captures upstream `expand_text_with_placeholders` output for a 2×4 grid byte-for-byte. We pass this fixture.
- `tests/fixtures/multi_image_ordering_proof.json` — Phase 0 fixture from real upstream output.
- Code comment in `src/chat_template.rs::build_image_block` cites upstream lines.

---

## REJECTED-2: "Patch vectors are HWC but config declares channels_first" (round 22)

### Claim
`flatten_to_patches`'s loop emits 16×16 patches as `(dy, dx, ch)` (HWC interleaved), but `preprocessor_config.json` declares `data_format: channels_first`. Therefore patches are fed to the encoder in the wrong layout.

### Why this is NOT a bug
The `data_format: channels_first` config refers to the **resized image format going into upstream's `convert_image_to_patches`** — i.e., torch tensor shape `(B, C, H, W)`. Inside that function, upstream PERMUTES to HWC before reshape:

```python
# upstream image_processing_lfm2_vl_fast.py:143-156 (convert_image_to_patches)
patched = images.reshape(B, C, n_h, ps, n_w, ps)
patched = patched.permute(0, 2, 4, 3, 5, 1)         # → (B, n_h, n_w, ps, ps, C)
patched = patched.reshape(B, n_h * n_w, -1)         # → (B, n_patches, ps*ps*C in HWC)
```

The final `.reshape(..., -1)` collapses `(ps, ps, C)` into 768 bytes in HWC order (last dim is C). So the actual ENCODER input IS HWC per-patch despite the upstream pipeline starting from a CHW image.

Our `(dy, dx, ch)` byte order in `flatten_to_patches` matches what upstream produces.

### Evidence
- `tests/fixtures/multi_image_ordering_proof.json` — captured from upstream and we match bit-for-bit.
- Code comment in `src/preproc/mod.rs::flatten_to_patches` (added round 22) cites upstream lines.
- Phase 0 G5 (resolved 2026-05-03) explicitly verified pixel layout.

---

## RESOLVED-1: "Resize uses image crate's Triangle filter, not torchvision bilinear+antialias" (round 42, fixed in same branch)

### Original claim
`flatten_to_patches` resized via `image::imageops::resize(..., FilterType::Triangle)`. Upstream `Lfm2VlImageProcessorFast` resizes via torchvision `F.resize(..., interpolation=BILINEAR, antialias=True)`. These are not the same algorithm — torchvision's antialiased bilinear runs a low-pass prefilter before sampling, while `image` crate's `Triangle` is a plain tent filter with no antialias prefilter. Cooperative callers feeding non-tile-aligned images would have gotten different `pixel_values` than upstream.

### Resolution
Replaced both `imageops::resize(..., FilterType::Triangle)` calls in `flatten_to_patches` (main resize and thumbnail resize) with a new `pil_bilinear_resize` helper backed by `fast_image_resize`'s `Convolution(FilterType::Bilinear)`. That's the PIL-compatible bilinear (Pillow's `Image.resize` with `Image.BILINEAR`), which is the exact target torchvision's `antialias=True` path was designed to match.

The `fast_image_resize` crate is widely used as the "PIL-parity" Rust resize and is what production VLM/image pipelines use for parity with HuggingFace processors.

### Why no `LFM_MODEL_PATH` parity script was required
The previous algorithm (`Triangle`) was definitively NOT what upstream uses. The new algorithm (PIL bilinear via `fast_image_resize`) IS what upstream uses by documented design (torchvision antialias=True ≡ PIL BILINEAR ≡ fast_image_resize Convolution(Bilinear)). Swapping one to the other replaces a known-divergent algorithm with the known-correct one — no real-model A/B is needed to know it's an improvement.

A real-model A/B is still warranted in v0.2 to confirm bit-exactness across all resize ratios (PIL-vs-fast_image_resize have rare 1-LSB differences for specific kernel-edge alignments), but the qualitative correctness fix is in place.

---

## REJECTED-3: "Padding to per-image max instead of upstream's fixed `max_num_patches`" (round 17)

### Claim
Upstream pads each tile to a fixed `max_num_patches = max(max_image_tokens × downsample_factor², (tile_size/patch_size)²) = 1024` for default budget. Our code pads only to per-image max (e.g., 256 for a 256-token single-tile image). Could silently change vision encoder outputs.

### Why this is INCONCLUSIVE — deferred, not rejected
Codex itself notes "the ONNX axis is dynamic so it may run". The vision encoder uses `pixel_attention_mask` to know which positions are real vs padded, so padding to a smaller size is more efficient and theoretically equivalent.

Confirming parity requires `LFM_MODEL_PATH` for a real-model side-by-side comparison. This is filed as v0.2 work. If a future review re-flags it with a concrete reproducer (real-model output diff), then it becomes a real bug; until then, it's an unproven theoretical concern.

### What would change my mind
A bug-finder script that:
1. Loads the real LFM2.5-VL ONNX vision encoder with `LFM_MODEL_PATH`
2. Runs a fixed test image through (a) our crate's preprocessing → vision encoder, and (b) upstream Python's preprocessing → same vision encoder
3. Compares the resulting `image_features` tensors element-wise
4. Shows non-zero diff > FP rounding tolerance

If such a script reproduces, this becomes a real bug.

---

## How to use this file

When you're about to file an adversarial-review finding, search this file first. If your finding matches a REJECTED entry:
- Read the cited evidence and verify it still holds against the current code
- If the evidence is no longer accurate (code changed, fixture changed, etc.), update the entry or file a new finding
- If the evidence still holds, do NOT re-flag — note in your review that you considered it and found it's already-debunked

If your finding is genuinely new: include enough detail (file path + line numbers, concrete reproducer or upstream-citation) that the next reviewer can verify in one pass.
