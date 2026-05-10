# Phase 0 verification scripts

Per spec `docs/superpowers/specs/2026-05-03-lfm-vlm-wrapper-design.md` §13.4,
these two scripts must run successfully and their JSON output must be
checked into `tests/fixtures/` **before any Rust code is written**. They
resolve verification gates G1–G6 — facts about the actual ONNX export
that the spec's `validate_*_session` validators and the §7 decode-loop
pseudocode currently treat as speculative.

## Setup

```sh
# Clone the model. Sized ~7 GB after LFS pull; takes a few minutes.
git clone https://huggingface.co/LiquidAI/LFM2.5-VL-450M-ONNX ./model

# Python deps:
pip install onnx onnxruntime numpy pillow transformers
```

## Gate A — `capture_onnx_io.py` (~5 seconds, no inference)

Captures input/output tensor names, dtypes, and shapes for the three ONNX
graphs. Resolves G1 (`position_ids` existence in decoder), G2 (conv-cache
naming), G3 (conv-cache layer indices), G4 (vision encoder output name),
G5 (decoder mask/position shape dynamicism).

```sh
LFM_HF_REVISION=$(git -C ./model rev-parse HEAD) \
    python3 ./lfm/scripts/capture_onnx_io.py \
    --onnx-dir ./model/onnx \
    > ./lfm/tests/fixtures/onnx_io_contract.json
```

## Gate B — `verify_multi_image_ordering.py` (~3-5 minutes, runs vision encoder)

Runs the vision encoder on three multi-image fixtures and verifies that
batched outputs preserve source-image order. Resolves G6.

```sh
LFM_HF_REVISION=$(git -C ./model rev-parse HEAD) \
    python3 ./lfm/scripts/verify_multi_image_ordering.py \
    --onnx ./model/onnx/vision_encoder_fp16.onnx \
    > ./lfm/tests/fixtures/multi_image_ordering_proof.json
```

If `all_passed: false`, the embed-merge invariant in spec §7 step 5 does
NOT hold for the current ONNX export — see the script's error output for
remediation options.

## After both scripts succeed

Both fixture JSONs MUST be committed to `tests/fixtures/` before Rust code
is written. The implementation plan (`superpowers:writing-plans` output)
treats these fixtures as a hard pre-flight gate.

The maintainer (or future automation, see spec §13.4 #18) re-runs both
scripts whenever LiquidAI publishes a new revision of the
`LFM2.5-VL-450M-ONNX` HF repo. The fixture-freshness loud-failure contract
in §13.4 protects against silent breakage at runtime, but the
re-capture step is a manual operation in v0.
