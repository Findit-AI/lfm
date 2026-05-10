#!/usr/bin/env python3
"""
Phase 0 Gate A — capture ONNX I/O contract.

Reads each LFM2.5-VL-450M-ONNX graph's input/output metadata (no inference,
no weights loaded) and writes a JSON fixture that the lfm Rust crate's
`validate_*_session` functions check against at session-build time.

Resolves verification gates G1–G5 (see spec §13.4):
  G1: decoder_model_merged.onnx position_ids input — present or not?
  G2: Conv-cache name convention — past_conv.{i} (dot) vs past_conv_{i}?
  G3: Conv-cache layer indices — compacted 0..9 or sparse with attn gaps?
  G4: Vision encoder output tensor name (literal string).
  G5: attention_mask / position_ids input shapes — static or dynamic?

USAGE
-----
First, clone the model repo:

    git clone https://huggingface.co/LiquidAI/LFM2.5-VL-450M-ONNX ./model

Then run from the model directory (or pass --onnx-dir):

    cd ./model/onnx
    LFM_HF_REVISION=$(git -C .. rev-parse HEAD) \\
        python3 /path/to/lfm/scripts/capture_onnx_io.py \\
        > /path/to/lfm/tests/fixtures/onnx_io_contract.json

Or with explicit path:

    LFM_HF_REVISION=$(git -C ./model rev-parse HEAD) \\
        python3 /path/to/lfm/scripts/capture_onnx_io.py \\
        --onnx-dir ./model/onnx \\
        > /path/to/lfm/tests/fixtures/onnx_io_contract.json

REQUIREMENTS
------------
    pip install onnx

(Note: this script does NOT require onnxruntime. It only reads protobuf
metadata. Total runtime: ~5 seconds.)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

try:
    import onnx
except ImportError:
    print(
        "error: onnx package not installed. Run `pip install onnx` first.",
        file=sys.stderr,
    )
    sys.exit(1)


# Files we capture. The variant suffix doesn't matter for I/O contract
# (all variants of the same graph have identical input/output names and
# shapes — only the weights differ), so we use the smallest available
# variant of each to minimize download time when the user only fetches
# a subset.
TARGET_FILES = [
    "vision_encoder.onnx",        # fp32 baseline
    "embed_tokens.onnx",          # fp32 baseline
    "decoder_model_merged.onnx",  # fp32 baseline
]


def shape_repr(tensor_type) -> list:
    """Convert protobuf shape to a JSON-serializable list.

    Each dim becomes either an int (concrete) or a string (dim_param,
    typically 'batch_size' or 'sequence_length').
    """
    return [
        d.dim_value if d.HasField("dim_value") else d.dim_param
        for d in tensor_type.shape.dim
    ]


# ONNX TensorProto element type → human-readable name.
# See: https://github.com/onnx/onnx/blob/main/onnx/onnx-ml.proto#L484
ELEM_NAMES = {
    1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16",
    6: "INT32", 7: "INT64", 8: "STRING", 9: "BOOL", 10: "FLOAT16",
    11: "DOUBLE", 12: "UINT32", 13: "UINT64", 14: "COMPLEX64",
    15: "COMPLEX128", 16: "BFLOAT16",
}


def describe_outlet(outlet) -> dict:
    """Convert one input/output ValueInfoProto to a JSON-friendly dict."""
    elem = outlet.type.tensor_type.elem_type
    return {
        "name": outlet.name,
        "dtype": ELEM_NAMES.get(elem, f"UNKNOWN({elem})"),
        "dtype_id": elem,
        "shape": shape_repr(outlet.type.tensor_type),
    }


def capture(onnx_dir: Path, hf_revision: str) -> dict:
    """Read each ONNX file and dump its I/O contract."""
    results = {
        "_metadata": {
            "captured_at": datetime.datetime.utcnow().isoformat() + "Z",
            "hf_repo": "LiquidAI/LFM2.5-VL-450M-ONNX",
            "hf_revision": hf_revision,
            "capture_script_version": "1.0",
        },
    }
    for fname in TARGET_FILES:
        path = onnx_dir / fname
        if not path.exists():
            # Try fp16 variant as fallback (only weights differ; I/O is identical).
            fp16 = onnx_dir / fname.replace(".onnx", "_fp16.onnx")
            if fp16.exists():
                print(
                    f"info: {fname} not found; using {fp16.name} (I/O contract is identical)",
                    file=sys.stderr,
                )
                path = fp16
            else:
                print(
                    f"error: neither {fname} nor {fp16.name} found in {onnx_dir}",
                    file=sys.stderr,
                )
                sys.exit(1)
        # load_external_data=False — we don't need the weights, just the graph metadata.
        # Skips the .onnx_data sidecar; runs in a few hundred ms even for the decoder.
        m = onnx.load(str(path), load_external_data=False)
        results[fname] = {
            "_source_path": str(path.name),
            "inputs": [describe_outlet(i) for i in m.graph.input],
            "outputs": [describe_outlet(o) for o in m.graph.output],
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Capture LFM2.5-VL-450M-ONNX I/O contract for the lfm Rust crate."
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the onnx files (default: current directory)",
    )
    args = parser.parse_args()

    hf_revision = os.environ.get("LFM_HF_REVISION", "unknown")
    if hf_revision == "unknown":
        print(
            "warning: LFM_HF_REVISION env var not set — fixture's _metadata.hf_revision\n"
            "         will be 'unknown', which will surface as an explicit warning in\n"
            "         validate_*_session error messages. Recommended:\n"
            "           LFM_HF_REVISION=$(git -C /path/to/model rev-parse HEAD) python3 ...",
            file=sys.stderr,
        )

    results = capture(args.onnx_dir, hf_revision)
    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
