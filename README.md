# 🌍 Multilingual ONNX Embedding Explorer

> **From production innovation to personal portfolio.**
> When there are no GPU access in your cloud data platform, how to use NLP use-cases then?
> I converted a multilingual embedding model to ONNX and shipped it working with engineering
> to production without any vendor support.
> This project recreates that same engineering — now across my two languages:
> **English** and **اردو (Urdu)**.

---

## The Problem Solution

When we started an AI-powered text project, there were some gpu issues and it wouldn't allow heavy Python dependencies like PyTorch or sentence-transformers access.

Standard solution? Didn't exist — the platform simply didn't support what we needed.

**Our solution:** Convert a 100+ language embedding model into **ONNX format** —
a universal, dependency-light ML model format that runs on pure CPU with no GPU required.
This let us deploy multilingual embeddings directly inside a User-Defined Function.

This was entirely a **Data Science team innovation**, not a vendor feature.

---

## This Project

A personal recreation using `paraphrase-multilingual-MiniLM-L12-v2` — a lighter
multilingual sentence transformer — with three components:

| File | Purpose |
|------|---------|
| `convert_to_onnx.py` | Converts the model from PyTorch → ONNX (FP32) |
| `benchmark.py` | Times CPU vs MPS (Apple M1 GPU) vs ONNX Runtime |
| `app.py` | Interactive Streamlit app with 3 tabs |

### What the Streamlit App Shows

**Tab 1 — Semantic Similarity:** Type any English sentence and its Urdu equivalent.
The ONNX model returns a cosine similarity score showing how close the meanings are
in the shared multilingual embedding space.

**Tab 2 — Embedding Clusters:** A UMAP visualisation showing English and Urdu sentences
about the same topic clustering together — proof that multilingual embeddings work
across languages.

**Tab 3 — Speed Benchmark:** Side-by-side inference latency comparison:
PyTorch CPU vs PyTorch MPS (Apple M1 GPU) vs ONNX Runtime.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/onnx-multilingual-portfolio
cd onnx-multilingual-portfolio

# Create environment (Apple Silicon)
conda create -n onnx-portfolio python=3.11 -y
conda activate onnx-portfolio

# Install dependencies
pip install -r requirements.txt

# Step 1: Convert model to ONNX (one-time, ~2 min)
python convert_to_onnx.py

# Step 2: Run benchmark
python benchmark.py

# Step 3: Launch app
streamlit run app.py
```

---

## Why ONNX?

| | PyTorch | ONNX Runtime |
|---|---------|-------------|
| Requires GPU | Optional | ❌ No |
| Heavy dependencies | ✅ Yes | ❌ No |
| Runs in restricted cloud UDFs | ❌ No | ✅ Yes |
| Inference speed (CPU) | Baseline | ~1.5–2x faster |
| Model portability | Limited | Universal |

---

## Tech Stack

- **Model:** `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages including Urdu)
- **Conversion:** Hugging Face `optimum` + `onnxruntime`
- **Inference:** ONNX Runtime (CPU), PyTorch MPS (M1 GPU)
- **Visualisation:** Streamlit, Plotly, UMAP
- **Hardware:** Apple M1 Pro, 16GB unified memory

---

## About

Built by **Bilal** — Data Scientist & ML Engineer.

Pakistani immigrant in America. I build things that work under constraints.

Connect: [LinkedIn](https://linkedin.com/in/muhammadbilaljamil)
