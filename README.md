# 🌍 Multilingual ONNX Embedding Explorer

> **Real engineering. Real constraints. Real production.**
> Enterprise ML pipelines often can't run heavy dependencies like PyTorch or
> load GPU-based models inside data platform UDFs.
> I solved this by converting a multilingual embedding model to ONNX —
> a format that runs fast on pure CPU, anywhere.
> This project recreates that same engineering across my two languages:
> **English** and **اردو (Urdu)**.

---

## The Problem

Enterprise data platforms are powerful for analytics — but they often impose
strict constraints on ML workloads: no GPU access, no heavy Python dependencies,
no way to load large PyTorch models inside user-defined functions (UDFs).

This is a common reality across the industry, not a unique edge case.

When our team needed multilingual text embeddings inside such an environment,
the standard path — load a sentence transformer, run inference — simply wasn't
available.

**The engineering question:** how do you bring state-of-the-art NLP into a
constrained, CPU-only environment with minimal dependencies?

---

## The Solution

**ONNX (Open Neural Network Exchange)** — a universal, open model format
that separates the model from its training framework.

By converting a 100+ language embedding model into ONNX format, we were able to:
- Run inference on **pure CPU** with no GPU required
- Eliminate heavy dependencies like PyTorch from the deployment environment
- Deploy multilingual embeddings directly inside platform UDFs
- Achieve **faster inference** than the original PyTorch model on CPU

This is now a pattern I apply beyond that single project — as this repo shows.

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
across languages and scripts.

**Tab 3 — Speed Benchmark:** Side-by-side inference latency comparison:
PyTorch CPU vs PyTorch MPS (Apple M1 GPU) vs ONNX Runtime.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/mbjamil92/onnx-multilingual-project
cd onnx-multilingual-project

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

Built by **Bilal** — Senior Data Scientist & ML Engineer.

Pakistani immigrant in America. I build things that work under constraints.

Connect: [LinkedIn](https://linkedin.com/in/muhammadbilaljamil)
