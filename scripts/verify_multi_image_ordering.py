#!/usr/bin/env python3
"""
Phase 0 Gate B — verify multi-image vision encoder ordering invariant.

Runs the actual vision_encoder.onnx on three multi-image cases and verifies
that for each, the concatenated batch produces image-embedding rows in
source-image order. This is the contract the Rust crate's embed-merge step
(spec §7 step 5) depends on; if the encoder reorders by tile-index across
images, every multi-image call silently corrupts.

Three cases (see spec §13.4):
  Case 1 (256² + 256²): single-tile-path multi-image — catches encoders
                        that reorder tiles within a batch.
  Case 2 (1024² + 1024²): multi-tile-path multi-image — catches encoders
                          that interleave tile rows across images.
  Case 3 (256² + 1024²): mixed-size cross-batch padding — catches encoders
                         that mishandle per-image image_max repadding.

Resolves verification gate G6.

USAGE
-----
First, clone the model AND run capture_onnx_io.py first (to know the output
tensor name):

    git clone https://huggingface.co/LiquidAI/LFM2.5-VL-450M-ONNX ./model
    LFM_HF_REVISION=$(git -C ./model rev-parse HEAD) \\
        python3 /path/to/lfm/scripts/capture_onnx_io.py \\
        --onnx-dir ./model/onnx \\
        > /path/to/lfm/tests/fixtures/onnx_io_contract.json

Then run Gate B (note: Phase 0 Gate A's output tells you the literal output
name; this script auto-detects from the ONNX graph if unspecified):

    LFM_HF_REVISION=$(git -C ./model rev-parse HEAD) \\
        python3 /path/to/lfm/scripts/verify_multi_image_ordering.py \\
        --onnx ./model/onnx/vision_encoder_fp16.onnx \\
        > /path/to/lfm/tests/fixtures/multi_image_ordering_proof.json

REQUIREMENTS
------------
    pip install onnx onnxruntime numpy pillow transformers

The transformers package is needed to use Lfm2VlImageProcessor for
preprocessing — we deliberately use the upstream processor here so
the test verifies the encoder, not our (yet-to-exist) Rust port.
Total runtime: ~3-5 minutes.

OUTPUT
------
Writes to stdout:

    {
      "_metadata": {...},
      "case_1_single_tile_pair": {
        "input_shapes": {...},
        "max_abs_diff_red": <float>,
        "max_abs_diff_blue": <float>,
        "tolerance": 5e-3,
        "passed": true
      },
      "case_2_multi_tile_pair": {...},
      "case_3_mixed_size_pair": {...},
      "all_passed": true
    }

If `all_passed: false`, the multi-image embed-merge invariant in §7 step 5
DOES NOT hold for this ONNX export. Implementation must either (a) special-case
to single-image-only generation, or (b) re-batch each image as its own
vision_encoder.run call (slower, but correctness over speed).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from transformers import AutoImageProcessor
except ImportError as e:
    print(
        f"error: missing dependency ({e}). Run:\n"
        f"  pip install onnx onnxruntime numpy pillow transformers",
        file=sys.stderr,
    )
    sys.exit(1)


# Tolerance for fp16 weights — see spec §13.4 Gate B.
# The fp16 vision encoder rounds to ~3 decimal places; allow up to 5e-3
# absolute difference in matched embeddings. Tighten to 1e-4 for fp32.
FP16_TOL = 5e-3
FP32_TOL = 1e-4


def detect_output_name(sess: ort.InferenceSession) -> str:
    """Pick the first output tensor name. Vision encoder has one output;
    if multiple, the first is conventionally the embeddings."""
    outs = sess.get_outputs()
    if not outs:
        raise RuntimeError("vision_encoder.onnx has no outputs")
    return outs[0].name


def make_solid_image(side: int, color: tuple[int, int, int]) -> Image.Image:
    """Build a solid-color RGB PIL image."""
    arr = np.full((side, side, 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def preprocess(processor, images: list[Image.Image]) -> dict[str, np.ndarray]:
    """Run upstream Lfm2VlImageProcessor and convert outputs to ONNX-ready
    numpy arrays. Lfm2VlImageProcessorFast only supports return_tensors="pt",
    so we convert through torch tensors → numpy."""
    out = processor(images=images, return_tensors="pt")
    # Transformers returns a BatchFeature dict; pick out the vision-encoder inputs.
    return {
        "pixel_values": out["pixel_values"].cpu().numpy().astype(np.float32),
        "pixel_attention_mask": out["pixel_attention_mask"].cpu().numpy().astype(np.int64),
        "spatial_shapes": out["spatial_shapes"].cpu().numpy().astype(np.int64),
    }


def run_case(
    sess: ort.InferenceSession,
    out_name: str,
    processor,
    name: str,
    img_a: Image.Image,
    img_b: Image.Image,
    tol: float,
) -> dict:
    """Run one Gate B case. Encode each image alone, then concatenated;
    verify the concat output equals the concatenation of the alone outputs."""
    inputs_a = preprocess(processor, [img_a])
    inputs_b = preprocess(processor, [img_b])
    inputs_ab = preprocess(processor, [img_a, img_b])

    embeds_a_alone = sess.run([out_name], inputs_a)[0]
    embeds_b_alone = sess.run([out_name], inputs_b)[0]
    embeds_ab_concat = sess.run([out_name], inputs_ab)[0]

    n_a = embeds_a_alone.shape[0]
    n_b = embeds_b_alone.shape[0]
    n_total = embeds_ab_concat.shape[0]
    if n_total != n_a + n_b:
        return {
            "case": name,
            "passed": False,
            "reason": f"shape mismatch: alone gave {n_a}+{n_b}={n_a+n_b} rows, "
            f"concat gave {n_total}",
            "input_shapes": {
                "img_a": img_a.size,
                "img_b": img_b.size,
                "n_a_alone": int(n_a),
                "n_b_alone": int(n_b),
                "n_concat": int(n_total),
            },
        }

    diff_a = float(np.abs(embeds_ab_concat[:n_a] - embeds_a_alone).max())
    diff_b = float(np.abs(embeds_ab_concat[n_a:] - embeds_b_alone).max())
    passed = diff_a <= tol and diff_b <= tol

    return {
        "case": name,
        "passed": passed,
        "input_shapes": {
            "img_a": list(img_a.size),
            "img_b": list(img_b.size),
            "n_a_alone": int(n_a),
            "n_b_alone": int(n_b),
            "n_concat": int(n_total),
        },
        "max_abs_diff_a": diff_a,
        "max_abs_diff_b": diff_b,
        "tolerance": tol,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify LFM2.5-VL multi-image vision encoder ordering invariant."
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Path to vision_encoder.onnx (or _fp16 / _q4 / _q8 variant). "
        "Tolerance auto-adjusts: fp32 → 1e-4, anything else → 5e-3.",
    )
    parser.add_argument(
        "--processor-id",
        default="LiquidAI/LFM2.5-VL-450M",
        help="HuggingFace processor model id (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.onnx.exists():
        print(f"error: {args.onnx} not found", file=sys.stderr)
        sys.exit(1)

    # Pick tolerance based on filename heuristic.
    # The fp32 baseline (no suffix) deserves the tighter tolerance.
    is_fp32 = args.onnx.stem == "vision_encoder"
    tol = FP32_TOL if is_fp32 else FP16_TOL

    print(f"info: loading {args.onnx} (tol={tol})", file=sys.stderr)
    sess = ort.InferenceSession(str(args.onnx))
    out_name = detect_output_name(sess)
    print(f"info: detected output tensor name = {out_name!r}", file=sys.stderr)

    print(f"info: loading image processor {args.processor_id}", file=sys.stderr)
    # Use AutoImageProcessor (not AutoProcessor) — for Gate B we only need
    # the image-side preprocessing; the tokenizer (TokenizersBackend) is a
    # transformers v5 feature and our v4.x install can't load it. Vision
    # encoder doesn't need tokens anyway.
    processor = AutoImageProcessor.from_pretrained(args.processor_id, trust_remote_code=True)

    # Build distinguishable solid-color images.
    red_sm = make_solid_image(256, (255, 0, 0))
    blue_sm = make_solid_image(256, (0, 0, 255))
    red_lg = make_solid_image(1024, (255, 0, 0))
    blue_lg = make_solid_image(1024, (0, 0, 255))

    print("info: running case 1 (256² + 256², single-tile path)", file=sys.stderr)
    case_1 = run_case(
        sess, out_name, processor,
        "case_1_single_tile_pair", red_sm, blue_sm, tol,
    )
    print("info: running case 2 (1024² + 1024², multi-tile path)", file=sys.stderr)
    case_2 = run_case(
        sess, out_name, processor,
        "case_2_multi_tile_pair", red_lg, blue_lg, tol,
    )
    print("info: running case 3 (256² + 1024², mixed-size cross-batch)", file=sys.stderr)
    case_3 = run_case(
        sess, out_name, processor,
        "case_3_mixed_size_pair", red_sm, blue_lg, tol,
    )

    hf_revision = os.environ.get("LFM_HF_REVISION", "unknown")
    if hf_revision == "unknown":
        print(
            "warning: LFM_HF_REVISION env var not set — _metadata.hf_revision\n"
            "         will be 'unknown'.",
            file=sys.stderr,
        )

    all_passed = case_1["passed"] and case_2["passed"] and case_3["passed"]

    result = {
        "_metadata": {
            "captured_at": datetime.datetime.utcnow().isoformat() + "Z",
            "hf_repo": "LiquidAI/LFM2.5-VL-450M-ONNX",
            "hf_revision": hf_revision,
            "verify_script_version": "1.0",
            "vision_encoder_path": str(args.onnx.name),
            "output_tensor_name": out_name,
            "tolerance": tol,
        },
        "case_1_single_tile_pair": case_1,
        "case_2_multi_tile_pair": case_2,
        "case_3_mixed_size_pair": case_3,
        "all_passed": all_passed,
    }

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if not all_passed:
        print(
            "\nFAIL: at least one Gate B case failed. The multi-image embed-merge\n"
            "invariant in §7 step 5 does NOT hold for this ONNX export. The Rust\n"
            "implementation must either (a) reject multi-image input (single-image\n"
            "only), or (b) re-batch each image as its own vision_encoder.run call\n"
            "(slower, but correct).",
            file=sys.stderr,
        )
        sys.exit(2)
    print("\nOK: all 3 Gate B cases passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
