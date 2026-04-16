"""
benchmark.py
============
Benchmarks embedding inference across 3 modes:
  1. PyTorch CPU   — baseline
  2. PyTorch MPS   — Apple M1 GPU (Metal Performance Shaders)
  3. ONNX Runtime  — the production-grade, dependency-light format

This is the core technical story:
  At work, we used ONNX because we had NO GPU in Snowflake.
  On an M1 Pro, we can now show ONNX is competitive even when a GPU exists.

Author: Bilal
"""

import time
import numpy as np
from pathlib import Path

MODEL_ID   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ONNX_DIR   = Path("models/multilingual-MiniLM-onnx")
N_RUNS     = 20   # number of inference runs per mode

# ── Test sentences (English + Urdu) ───────────────────────────────────────────
TEST_SENTENCES = [
    # English
    "I am a data scientist building machine learning models.",
    "Natural language processing helps computers understand human text.",
    "The weather today is sunny and warm.",
    # Urdu (romanized + native script)
    "میں ایک ڈیٹا سائنٹسٹ ہوں جو مشین لرننگ ماڈل بناتا ہوں۔",
    "قدرتی زبان کی پروسیسنگ کمپیوٹر کو انسانی متن سمجھنے میں مدد کرتی ہے۔",
    "آج موسم دھوپ اور گرم ہے۔",
]


def mean_pooling(token_embeddings, attention_mask):
    """Average token embeddings weighted by attention mask."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def benchmark_pytorch_cpu(sentences, n_runs):
    import torch
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F

    print("\n[1/3] PyTorch CPU")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    inputs = tokenizer(sentences, padding=True, truncation=True,
                       return_tensors="pt", max_length=128)

    # Warm up
    with torch.no_grad():
        _ = model(**inputs)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = model(**inputs)
            emb = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    print(f"    Avg latency: {avg_ms:.1f} ms  (over {n_runs} runs)")
    return avg_ms, emb.numpy()


def benchmark_pytorch_mps(sentences, n_runs):
    import torch
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F

    if not torch.backends.mps.is_available():
        print("\n[2/3] PyTorch MPS — SKIPPED (not available)")
        return None, None

    print("\n[2/3] PyTorch MPS (Apple M1 GPU)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to("mps")
    model.eval()

    inputs = tokenizer(sentences, padding=True, truncation=True,
                       return_tensors="pt", max_length=128)
    inputs = {k: v.to("mps") for k, v in inputs.items()}

    # Warm up
    with torch.no_grad():
        _ = model(**inputs)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = model(**inputs)
            emb = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            torch.mps.synchronize()   # wait for GPU to finish
            times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    print(f"    Avg latency: {avg_ms:.1f} ms  (over {n_runs} runs)")
    return avg_ms, emb.cpu().numpy()


def benchmark_onnx(sentences, n_runs):
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    import torch
    import torch.nn.functional as F

    if not ONNX_DIR.exists():
        print("\n[3/3] ONNX Runtime — SKIPPED (run convert_to_onnx.py first)")
        return None, None

    print("\n[3/3] ONNX Runtime (CPU execution provider)")
    tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
    model = ORTModelForFeatureExtraction.from_pretrained(ONNX_DIR)

    inputs = tokenizer(sentences, padding=True, truncation=True,
                       return_tensors="pt", max_length=128)

    # Warm up
    _ = model(**inputs)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = model(**inputs)
        emb = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    print(f"    Avg latency: {avg_ms:.1f} ms  (over {n_runs} runs)")
    return avg_ms, emb.detach().numpy()


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run():
    print("=" * 60)
    print("  INFERENCE BENCHMARK")
    print("  Sentences: English + Urdu (6 total)")
    print(f"  Runs per mode: {N_RUNS}")
    print("=" * 60)

    cpu_ms,  cpu_emb  = benchmark_pytorch_cpu(TEST_SENTENCES, N_RUNS)
    mps_ms,  mps_emb  = benchmark_pytorch_mps(TEST_SENTENCES, N_RUNS)
    onnx_ms, onnx_emb = benchmark_onnx(TEST_SENTENCES, N_RUNS)

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Mode':<20} {'Avg Latency':>15}  {'vs CPU':>10}")
    print(f"  {'-'*20} {'-'*15}  {'-'*10}")
    print(f"  {'PyTorch CPU':<20} {cpu_ms:>12.1f} ms  {'(baseline)':>10}")
    if mps_ms:
        speedup = cpu_ms / mps_ms
        print(f"  {'PyTorch MPS (M1)':<20} {mps_ms:>12.1f} ms  {speedup:>9.1f}x")
    if onnx_ms:
        speedup = cpu_ms / onnx_ms
        print(f"  {'ONNX Runtime':<20} {onnx_ms:>12.1f} ms  {speedup:>9.1f}x")

    # ── Cross-lingual similarity check ────────────────────────────────────────
    if onnx_emb is not None:
        print("\n  CROSS-LINGUAL SIMILARITY (English ↔ Urdu)")
        print("  Using ONNX embeddings")
        print(f"  {'-'*55}")
        pairs = [
            (0, 3, "Data scientist sentence"),
            (1, 4, "NLP sentence"),
            (2, 5, "Weather sentence"),
        ]
        for i, j, label in pairs:
            sim = cosine_similarity(onnx_emb[i], onnx_emb[j])
            bar = "█" * int(sim * 20)
            print(f"  {label:<25} sim={sim:.3f}  {bar}")

    print("\n  ✅ Benchmark complete!")
    print("  Next: run `streamlit run app.py` for the interactive demo")
    print("=" * 60)


if __name__ == "__main__":
    run()
