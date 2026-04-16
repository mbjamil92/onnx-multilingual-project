"""
app.py  —  Multilingual ONNX Embedding Explorer
================================================
An interactive Streamlit app showcasing:
  Tab 1: Semantic similarity across English & Urdu
  Tab 2: Embedding cluster visualisation (UMAP)
  Tab 3: Live inference speed comparison (CPU vs MPS vs ONNX)

Author: Bilal
"""

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual ONNX Embeddings",
    page_icon="🌍",
    layout="wide",
)

ONNX_DIR = Path("models/multilingual-MiniLM-onnx")
MODEL_ID  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Cached model loaders ──────────────────────────────────────────────────────
@st.cache_resource
def load_onnx_model():
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
    model = ORTModelForFeatureExtraction.from_pretrained(ONNX_DIR)
    return tokenizer, model

@st.cache_resource
def load_pytorch_model(device="cpu"):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return tokenizer, model


def mean_pool(token_emb, attn_mask):
    import torch
    import torch.nn.functional as F
    mask = attn_mask.unsqueeze(-1).expand(token_emb.size()).float()
    emb = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(emb, p=2, dim=1)


def get_onnx_embeddings(sentences):
    import torch, torch.nn.functional as F
    tokenizer, model = load_onnx_model()
    inputs = tokenizer(sentences, padding=True, truncation=True,
                       return_tensors="pt", max_length=128)
    with torch.no_grad():
        out = model(**inputs)
    emb = mean_pool(out.last_hidden_state, inputs["attention_mask"])
    return emb.detach().numpy()


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌍 Multilingual ONNX Embedding Explorer")
st.markdown("""
A personal ML engineering project exploring **multilingual sentence embeddings** via ONNX —
a universal model format that runs fast on CPU with no GPU required.

Built across two of my languages: **English** and **اردو (Urdu)**.
""")
st.divider()

# Check ONNX model exists
if not ONNX_DIR.exists():
    st.error("⚠️ ONNX model not found. Run `python convert_to_onnx.py` first.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "🔤 Semantic Similarity",
    "🗺️ Embedding Clusters",
    "⚡ Speed Benchmark"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Semantic Similarity
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Semantic Similarity — English ↔ Urdu")
    st.markdown("""
    Type any sentence in English and its Urdu equivalent (or any two sentences).
    The model produces embeddings in a **shared multilingual space** — meaning
    semantically similar sentences land close together, regardless of language.
    """)

    col1, col2 = st.columns(2)
    with col1:
        sent_en = st.text_area(
            "English sentence",
            value="I am a data scientist building machine learning models.",
            height=100,
        )
    with col2:
        sent_ur = st.text_area(
            "Urdu sentence (or any other language)",
            value="میں ایک ڈیٹا سائنٹسٹ ہوں جو مشین لرننگ ماڈل بناتا ہوں۔",
            height=100,
        )

    if st.button("Compute Similarity", type="primary"):
        with st.spinner("Running ONNX inference..."):
            t0 = time.perf_counter()
            embs = get_onnx_embeddings([sent_en, sent_ur])
            elapsed_ms = (time.perf_counter() - t0) * 1000

        sim = cosine_sim(embs[0], embs[1])
        pct = sim * 100

        st.metric("Cosine Similarity", f"{sim:.4f}", f"{pct:.1f}% semantic overlap")
        st.progress(min(sim, 1.0))
        st.caption(f"⚡ ONNX inference time: {elapsed_ms:.1f} ms")

        if sim > 0.85:
            st.success("✅ Very high similarity — the model understands these are the same meaning across languages!")
        elif sim > 0.60:
            st.info("🔵 Moderate similarity — related concepts but not identical meaning.")
        else:
            st.warning("🟡 Low similarity — the sentences have different meanings.")

    st.divider()
    st.subheader("Pre-built Comparison Set")
    st.markdown("See how the model handles a set of English/Urdu translation pairs:")

    preset_pairs = [
        ("I love machine learning.", "مجھے مشین لرننگ سے محبت ہے۔"),
        ("The weather is cold today.", "آج موسم ٹھنڈا ہے۔"),
        ("Pakistan is a beautiful country.", "پاکستان ایک خوبصورت ملک ہے۔"),
        ("I miss my family back home.", "مجھے گھر پر اپنے خاندان کی یاد آتی ہے۔"),
        ("Data science is the future.", "ڈیٹا سائنس مستقبل ہے۔"),
    ]

    if st.button("Run all pairs"):
        all_sents = [s for pair in preset_pairs for s in pair]
        with st.spinner("Computing embeddings for all pairs..."):
            embs = get_onnx_embeddings(all_sents)

        results = []
        for i, (en, ur) in enumerate(preset_pairs):
            sim = cosine_sim(embs[i*2], embs[i*2+1])
            results.append({"English": en, "Urdu": ur, "Similarity": round(sim, 4)})

        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(
            df.style.background_gradient(subset=["Similarity"], cmap="Greens"),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Embedding Clusters
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Embedding Cluster Visualisation")
    st.markdown("""
    Each sentence gets a **384-dimensional** embedding vector. We reduce this to 2D
    using UMAP so we can see clusters. The key insight: **English and Urdu
    translations land near each other** in the shared semantic space.
    """)

    corpus = {
        "Technology (EN)": [
            "Machine learning model training",
            "Deep neural network architecture",
            "Natural language processing pipeline",
            "Data science and analytics",
            "Artificial intelligence research",
        ],
        "Technology (UR)": [
            "مشین لرننگ ماڈل ٹریننگ",
            "گہرے نیورل نیٹ ورک کا ڈھانچہ",
            "قدرتی زبان کی پروسیسنگ پائپ لائن",
            "ڈیٹا سائنس اور تجزیہ",
            "مصنوعی ذہانت کی تحقیق",
        ],
        "Pakistan (EN)": [
            "Lahore is a historic city in Pakistan",
            "Pakistani food is very flavorful",
            "The mountains of northern Pakistan are beautiful",
            "Cricket is the most popular sport in Pakistan",
            "Urdu is the national language of Pakistan",
        ],
        "Pakistan (UR)": [
            "لاہور پاکستان کا ایک تاریخی شہر ہے",
            "پاکستانی کھانا بہت لذیذ ہے",
            "پاکستان کے شمالی پہاڑ خوبصورت ہیں",
            "کرکٹ پاکستان کا مقبول ترین کھیل ہے",
            "اردو پاکستان کی قومی زبان ہے",
        ],
        "Weather (EN)": [
            "It is sunny and warm today",
            "Heavy rain is expected tomorrow",
            "Cold winter mornings in December",
        ],
        "Weather (UR)": [
            "آج دھوپ اور گرمی ہے",
            "کل بھاری بارش متوقع ہے",
            "دسمبر میں ٹھنڈی سردیوں کی صبحیں",
        ],
    }

    if st.button("Generate Cluster Plot", type="primary"):
        all_sents, labels, groups = [], [], []
        for group, sents in corpus.items():
            for s in sents:
                all_sents.append(s)
                labels.append(s[:40] + "..." if len(s) > 40 else s)
                groups.append(group)

        with st.spinner("Computing embeddings and running UMAP..."):
            embs = get_onnx_embeddings(all_sents)

            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5)
                coords = reducer.fit_transform(embs)
            except ImportError:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=5)
                coords = reducer.fit_transform(embs)
                st.caption("(Using t-SNE — install umap-learn for UMAP)")

        import pandas as pd
        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "label": labels,
            "group": groups,
        })

        fig = px.scatter(
            df, x="x", y="y", color="group", hover_data=["label"],
            title="Multilingual Embedding Space — English & Urdu sentences cluster by meaning",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(marker=dict(size=10, opacity=0.85))
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        💡 Notice how English and Urdu sentences about the **same topic**
        cluster together — even though they use completely different scripts.
        This is the power of multilingual embeddings.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Speed Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Inference Speed: CPU vs MPS vs ONNX")
    st.markdown("""
    Benchmarking three inference modes on an **Apple M1 Pro**:
    - **CPU** — PyTorch, no hardware acceleration
    - **MPS** — PyTorch on the M1 GPU via Metal Performance Shaders
    - **ONNX Runtime** — portable, dependency-light, CPU optimised
    """)

    n_runs = st.slider("Number of inference runs", 5, 50, 20)
    test_sents = [
        "I am a data scientist building machine learning models.",
        "میں ایک ڈیٹا سائنٹسٹ ہوں جو مشین لرننگ ماڈل بناتا ہوں۔",
        "Natural language processing helps computers understand text.",
        "قدرتی زبان کی پروسیسنگ کمپیوٹر کو متن سمجھنے میں مدد کرتی ہے۔",
    ]

    if st.button("Run Benchmark", type="primary"):
        results = {}

        # CPU
        with st.spinner("Benchmarking PyTorch CPU..."):
            import torch, torch.nn.functional as F
            tok, mdl = load_pytorch_model("cpu")
            inputs = tok(test_sents, padding=True, truncation=True,
                         return_tensors="pt", max_length=128)
            with torch.no_grad():
                _ = mdl(**inputs)  # warm up
            times = []
            with torch.no_grad():
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    out = mdl(**inputs)
                    mean_pool(out.last_hidden_state, inputs["attention_mask"])
                    times.append(time.perf_counter() - t0)
            results["PyTorch CPU"] = np.mean(times) * 1000

        # MPS
        if torch.backends.mps.is_available():
            with st.spinner("Benchmarking PyTorch MPS (M1 GPU)..."):
                tok_mps, mdl_mps = load_pytorch_model("mps")
                inp_mps = {k: v.to("mps") for k, v in inputs.items()}
                with torch.no_grad():
                    _ = mdl_mps(**inp_mps)
                times = []
                with torch.no_grad():
                    for _ in range(n_runs):
                        t0 = time.perf_counter()
                        out = mdl_mps(**inp_mps)
                        mean_pool(out.last_hidden_state, inp_mps["attention_mask"])
                        torch.mps.synchronize()
                        times.append(time.perf_counter() - t0)
                results["PyTorch MPS (M1)"] = np.mean(times) * 1000

        # ONNX
        with st.spinner("Benchmarking ONNX Runtime..."):
            tok_onnx, mdl_onnx = load_onnx_model()
            inp_onnx = tok_onnx(test_sents, padding=True, truncation=True,
                                return_tensors="pt", max_length=128)
            _ = mdl_onnx(**inp_onnx)  # warm up
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                out = mdl_onnx(**inp_onnx)
                mean_pool(out.last_hidden_state, inp_onnx["attention_mask"])
                times.append(time.perf_counter() - t0)
            results["ONNX Runtime"] = np.mean(times) * 1000

        # ── Plot ──────────────────────────────────────────────────────────────
        import pandas as pd
        df = pd.DataFrame(
            list(results.items()), columns=["Mode", "Avg Latency (ms)"]
        )
        cpu_ms = results["PyTorch CPU"]
        df["Speedup vs CPU"] = (cpu_ms / df["Avg Latency (ms)"]).round(2)

        colors = {"PyTorch CPU": "#EF553B", "PyTorch MPS (M1)": "#00CC96",
                  "ONNX Runtime": "#636EFA"}
        fig = go.Figure([
            go.Bar(
                x=df["Mode"],
                y=df["Avg Latency (ms)"],
                marker_color=[colors.get(m, "#888") for m in df["Mode"]],
                text=df["Avg Latency (ms)"].apply(lambda x: f"{x:.1f} ms"),
                textposition="outside",
            )
        ])
        fig.update_layout(
            title=f"Average Inference Latency ({n_runs} runs, {len(test_sents)} sentences)",
            yaxis_title="Latency (ms) — lower is better",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.set_index("Mode"), use_container_width=True)

        best = df.loc[df["Avg Latency (ms)"].idxmin(), "Mode"]
        st.success(f"🏆 Fastest mode: **{best}**")

        st.info("""
        **Why ONNX?** ONNX Runtime runs purely on CPU — no PyTorch, no CUDA,
        no GPU required. This makes it ideal for deploying in constrained
        environments where heavy ML dependencies can't be installed.
        Even on an M1 Pro, ONNX is competitive and far more portable
        than either PyTorch mode.
        """)
