"""
app.py — Fake News Detector Streamlit Web App.

A polished, visually striking web interface that lets users paste a news
article or headline and instantly classify it as Real or Fake using a
pre-trained PassiveAggressiveClassifier with TF-IDF features.

Usage:
    streamlit run app.py
"""

import os
import json
import time

import streamlit as st
import joblib

from .utils import preprocess_text


# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "pac_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")


# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Import premium font ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ──────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* ── Hero section ────────────────────────────────────────── */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff 0%, #7b2ff7 50%, #ff0080 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #a0aec0;
        font-weight: 300;
        margin-bottom: 1.5rem;
    }

    /* ── Glassmorphism card ───────────────────────────────────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .glass-card-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Result banners ──────────────────────────────────────── */
    .result-banner {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        animation: slideUp 0.5s ease-out;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .result-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .result-real .result-icon { font-size: 3rem; }
    .result-real .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981;
        margin: 0.5rem 0;
    }
    .result-real .result-desc {
        color: #6ee7b7;
        font-size: 1rem;
        font-weight: 300;
    }

    .result-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.08) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .result-fake .result-icon { font-size: 3rem; }
    .result-fake .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ef4444;
        margin: 0.5rem 0;
    }
    .result-fake .result-desc {
        color: #fca5a5;
        font-size: 1rem;
        font-weight: 300;
    }

    /* ── Confidence meter ────────────────────────────────────── */
    .confidence-container {
        margin: 1rem 0;
    }

    .confidence-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }

    .confidence-bar-bg {
        width: 100%;
        height: 10px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        overflow: hidden;
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s ease-in-out;
    }

    .confidence-value {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-top: 0.5rem;
    }

    /* ── Stats pills ─────────────────────────────────────────── */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin: 1rem 0;
    }

    .stat-pill {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }

    /* ── Sidebar styling ─────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
    }

    .sidebar-section {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }

    .sidebar-section-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #7b2ff7;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.6rem;
    }

    .sidebar-text {
        font-size: 0.85rem;
        color: #a0aec0;
        line-height: 1.6;
    }

    /* ── History list ────────────────────────────────────────── */
    .history-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .history-text {
        font-size: 0.8rem;
        color: #cbd5e0;
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        margin-right: 0.8rem;
    }

    .history-badge {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        white-space: nowrap;
    }

    .badge-real {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-fake {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ── Button override ─────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #7b2ff7 0%, #00d2ff 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(123, 47, 247, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(123, 47, 247, 0.5);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* ── Text area override ──────────────────────────────────── */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(123, 47, 247, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(123, 47, 247, 0.15) !important;
    }

    .stTextArea textarea::placeholder {
        color: #4a5568 !important;
    }

    /* ── Divider ─────────────────────────────────────────────── */
    .subtle-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        margin: 1.5rem 0;
    }

    /* ── Sample cards ────────────────────────────────────────── */
    .sample-header {
        font-size: 0.85rem;
        font-weight: 500;
        color: #718096;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load the trained model, vectorizer, and training metadata."""
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(VECTORIZER_PATH):
        return None, None, None

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    metadata = None
    if os.path.isfile(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

    return model, vectorizer, metadata


model, vectorizer, metadata = load_model()


# ─── Session State ───────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size: 2.5rem;">🔍</span>
        <div style="font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin-top: 0.3rem;">
            Fake News Detector
        </div>
        <div style="font-size: 0.75rem; color: #718096;">Powered by Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

    # How It Works
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">🧠 How It Works</div>
        <div class="sidebar-text">
            <strong>1. Text Preprocessing</strong><br>
            Your article is cleaned — lowercased, stripped of punctuation, 
            and common filler words are removed.<br><br>
            <strong>2. TF-IDF Vectorization</strong><br>
            The cleaned text is converted into a numerical matrix that 
            highlights meaningful, unique words.<br><br>
            <strong>3. Classification</strong><br>
            A PassiveAggressive Classifier analyzes the patterns and 
            predicts whether the article is real or fake.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model Stats
    if metadata:
        acc = metadata.get("accuracy", 0)
        total = metadata.get("total_samples", 0)
        features = metadata.get("max_features", 0)

        st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">📊 Model Performance</div>
            <div class="stats-grid">
                <div class="stat-pill">
                    <div class="stat-value">{acc}%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-value">{total:,}</div>
                    <div class="stat-label">Articles</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-value">{features:,}</div>
                    <div class="stat-label">Features</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # About
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">ℹ️ About</div>
        <div class="sidebar-text">
            Built with Streamlit, scikit-learn, and NLTK.<br>
            Dataset: Kaggle Fake & Real News Dataset.<br><br>
            <em>This tool is for educational purposes and should 
            not be the sole basis for determining news authenticity.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Section ────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-container">
    <div class="hero-title">Fake News Detector</div>
    <div class="hero-subtitle">
        Paste any news article below and let AI determine its authenticity in seconds
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Model Not Found Guard ──────────────────────────────────────────────────

if model is None or vectorizer is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
        <div style="font-size: 1.3rem; font-weight: 600; color: #fbbf24; margin-bottom: 0.5rem;">
            Model Not Found
        </div>
        <div style="color: #a0aec0; font-size: 0.95rem; line-height: 1.7;">
            The trained model hasn't been generated yet. Please run:<br><br>
            <code style="background: rgba(123,47,247,0.15); padding: 0.4rem 1rem; border-radius: 8px; color: #c4b5fd;">
                python train_model.py
            </code>
            <br><br>
            Make sure <strong>True.csv</strong> and <strong>Fake.csv</strong> from Kaggle 
            are placed in the <strong>data/</strong> folder first.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Input Section ───────────────────────────────────────────────────────────

st.markdown("""
<div class="glass-card">
    <div class="glass-card-header">
        📰 Paste a News Article or Headline
    </div>
</div>
""", unsafe_allow_html=True)

article_text = st.text_area(
    label="Enter article text",
    placeholder="Paste a news article, headline, or any text you want to verify...",
    height=180,
    label_visibility="collapsed",
)

# ─── Sample Articles ─────────────────────────────────────────────────────────

st.markdown('<div class="sample-header">💡 Try a sample:</div>', unsafe_allow_html=True)

sample_cols = st.columns(2)

SAMPLE_REAL = (
    "The Federal Reserve announced a quarter-point interest rate increase "
    "on Wednesday, citing continued strength in the labor market and steady "
    "economic growth. Chair Jerome Powell stated that future decisions will "
    "remain data-dependent, with inflation still above the 2% target."
)

SAMPLE_FAKE = (
    "BREAKING: Scientists confirm the moon is actually made of compressed "
    "cheese deposits. NASA has been covering this up for decades according "
    "to leaked documents. The government plans to mine the moon for its "
    "high-quality mozzarella reserves by 2025."
)

with sample_cols[0]:
    if st.button("✅ Sample Real News", key="sample_real", use_container_width=True):
        st.session_state["sample_text"] = SAMPLE_REAL
        st.rerun()

with sample_cols[1]:
    if st.button("🚨 Sample Fake News", key="sample_fake", use_container_width=True):
        st.session_state["sample_text"] = SAMPLE_FAKE
        st.rerun()

# Apply sample text if set
if "sample_text" in st.session_state and not article_text:
    article_text = st.session_state.pop("sample_text")

st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)


# ─── Prediction ──────────────────────────────────────────────────────────────

analyze_clicked = st.button("🔍  Analyze Article", key="analyze_btn", use_container_width=True)

if analyze_clicked:
    if not article_text or not article_text.strip():
        st.warning("⚠️ Please paste a news article or headline to analyze.")
    else:
        # Progress animation
        progress_bar = st.progress(0)
        status_text = st.empty()

        steps = [
            ("🧹 Cleaning text...", 25),
            ("📊 Extracting features...", 50),
            ("🧠 Running classifier...", 75),
            ("✨ Generating report...", 100),
        ]

        for msg, pct in steps:
            status_text.markdown(
                f'<div style="color: #a0aec0; font-size: 0.85rem; text-align: center;">{msg}</div>',
                unsafe_allow_html=True,
            )
            progress_bar.progress(pct)
            time.sleep(0.3)

        progress_bar.empty()
        status_text.empty()

        # ── Actual prediction ──
        clean = preprocess_text(article_text)
        tfidf_vector = vectorizer.transform([clean])
        prediction = model.predict(tfidf_vector)[0]

        # Confidence via decision_function (distance from hyperplane)
        decision_score = model.decision_function(tfidf_vector)[0]
        confidence = min(abs(decision_score) * 30, 99.9)  # Scale to 0-100 range

        # ── Display result ──
        if prediction == 1:
            # REAL
            st.markdown(f"""
            <div class="result-banner result-real">
                <div class="result-icon">✅</div>
                <div class="result-label">Looks Authentic!</div>
                <div class="result-desc">
                    Our model believes this article appears to be legitimate news.
                </div>
                <div class="confidence-container" style="margin-top: 1.2rem;">
                    <div class="confidence-label" style="text-align: center;">Confidence Level</div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width: {confidence:.0f}%; background: linear-gradient(90deg, #10b981, #34d399);"></div>
                    </div>
                    <div class="confidence-value" style="color: #10b981;">{confidence:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            label_str = "REAL"

        else:
            # FAKE
            st.markdown(f"""
            <div class="result-banner result-fake">
                <div class="result-icon">🚨</div>
                <div class="result-label">Warning: This Looks Like Fake News!</div>
                <div class="result-desc">
                    Our model has flagged this article as potentially fabricated or misleading.
                </div>
                <div class="confidence-container" style="margin-top: 1.2rem;">
                    <div class="confidence-label" style="text-align: center;">Confidence Level</div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width: {confidence:.0f}%; background: linear-gradient(90deg, #ef4444, #f87171);"></div>
                    </div>
                    <div class="confidence-value" style="color: #ef4444;">{confidence:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            label_str = "FAKE"

        # ── Save to history ──
        snippet = article_text[:80] + ("..." if len(article_text) > 80 else "")
        st.session_state.history.insert(0, {"text": snippet, "label": label_str})
        # Keep last 10
        st.session_state.history = st.session_state.history[:10]


# ─── Prediction History ─────────────────────────────────────────────────────

if st.session_state.history:
    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-header">📋 Recent Predictions</div>
    """, unsafe_allow_html=True)

    for item in st.session_state.history:
        badge_class = "badge-real" if item["label"] == "REAL" else "badge-fake"
        badge_text = "✅ Real" if item["label"] == "REAL" else "🚨 Fake"

        st.markdown(f"""
        <div class="history-item">
            <div class="history-text">{item['text']}</div>
            <span class="history-badge {badge_class}">{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem; color: #4a5568; font-size: 0.75rem;">
    Built with ❤️ using Streamlit &nbsp;•&nbsp; PassiveAggressive Classifier &nbsp;•&nbsp; TF-IDF
    <br>
    <span style="color: #2d3748;">For educational purposes only — always verify news from multiple sources.</span>
</div>
""", unsafe_allow_html=True)
