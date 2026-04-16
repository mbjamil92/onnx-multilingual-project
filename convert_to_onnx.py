"""
convert_to_onnx.py
==================
Converts paraphrase-multilingual-MiniLM-L12-v2 (a sentence transformer)
into ONNX format for fast, portable CPU inference.

Why ONNX?
---------
At work, we had no GPU access in Snowflake (GCP). PyTorch/sentence-transformers
couldn't be installed in PySpark. By converting LaBSE to ONNX, we brought
multilingual embeddings into production without any GPU or heavy dependencies.

This script recreates the same engineering innovation — now on a personal project
using a lighter multilingual model, on Apple Silicon (M1 Pro).

Author: Bilal
"""

import os
import time
from pathlib import Path

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_ID   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = Path("models/multilingual-MiniLM-onnx")

def convert():
    print("=" * 60)
    print("  ONNX Model Conversion")
    print("  Model : paraphrase-multilingual-MiniLM-L12-v2")
    print("  Format: FP32 (same approach used in production at work)")
    print("=" * 60)

    # ── Step 1: Load via Optimum and export to ONNX ───────────────────────────
    print("\n[1/3] Loading model and exporting to ONNX (FP32)...")
    start = time.time()

    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_ID,
        export=True,          # triggers PyTorch → ONNX conversion
        # No fp16 flag = FP32 by default (same as production Snowflake UDF)
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    elapsed = time.time() - start
    print(f"    ✅ Model exported in {elapsed:.1f}s")

    # ── Step 2: Save model + tokenizer locally ────────────────────────────────
    print(f"\n[2/3] Saving ONNX model to {OUTPUT_DIR} ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"    ✅ Saved successfully")

    # ── Step 3: Inspect the ONNX model ───────────────────────────────────────
    print("\n[3/3] Inspecting ONNX model inputs/outputs...")
    import onnx
    onnx_path = OUTPUT_DIR / "model.onnx"
    onnx_model = onnx.load(str(onnx_path))

    print("\n  ONNX Model Input Details:")
    for inp in onnx_model.graph.input:
        ttype = inp.type.tensor_type.elem_type
        dtype_map = {1: "tensor(float) = FP32", 7: "tensor(int64)"}
        print(f"    Name: {inp.name:<25} Type: {dtype_map.get(ttype, ttype)}")

    print("\n  ONNX Model Output Details:")
    for out in onnx_model.graph.output:
        ttype = out.type.tensor_type.elem_type
        dtype_map = {1: "tensor(float) = FP32", 7: "tensor(int64)"}
        print(f"    Name: {out.name:<25} Type: {dtype_map.get(ttype, ttype)}")

    # File size
    size_mb = onnx_path.stat().st_size / 1_000_000
    print(f"\n  Model size: {size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("  ✅ ONNX conversion complete!")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print("  Next: run benchmark.py to compare CPU vs MPS vs ONNX")
    print("=" * 60)


if __name__ == "__main__":
    convert()
