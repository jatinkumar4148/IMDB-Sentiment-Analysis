# ============================================================
# Sentiment Analysis App — Streamlit
# Model: Logistic Regression (IMDB Movie Reviews)
# ============================================================

import streamlit as st
import pickle
import re

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="🎬 Sentiment Analysis App",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# Custom CSS — Blue Theme + Animations
# ============================================================
st.markdown("""
<style>
    /* Remove White Space Top */
    header {
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .stApp > header {
        background: transparent !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stHeader"] {
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .block-container {
        padding-top: 1rem !important;
    }

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0a0a2e 0%, #0d1b4b 50%, #0a0a2e 100%);
        color: #f5f5f5;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #050520 0%, #0a0a3e 100%);
        border-right: 1px solid #FFD700;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFD700 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li {
        color: #aaaacc !important;
    }

    /* Header Animation */
    @keyframes fadeDown {
        from { opacity: 0; transform: translateY(-30px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        animation: fadeDown 0.8s ease forwards;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FFD700;
        letter-spacing: 3px;
        margin-bottom: 0.3rem;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
    }
    .main-header p {
        color: #7788cc;
        font-size: 0.95rem;
        letter-spacing: 2px;
    }

    /* Gold Divider */
    .gold-divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #FFD700, transparent);
        margin: 1rem 0 2rem 0;
    }

    /* About Button */
    .about-btn {
        position: fixed;
        top: 1rem;
        right: 1rem;
        background: #FFD700;
        color: #0a0a2e;
        border: none;
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 1px;
        cursor: pointer;
        z-index: 999;
        transition: all 0.3s ease;
    }
    .about-btn:hover {
        background: #FFC200;
        transform: scale(1.05);
    }

    /* Input Labels */
    label {
        color: #FFD700 !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        font-size: 0.85rem !important;
    }

    /* Input Fields */
    .stTextInput input {
        background-color: #0d1440 !important;
        border: 1px solid #FFD700 !important;
        border-radius: 8px !important;
        color: #f5f5f5 !important;
        padding: 12px !important;
    }
    .stTextInput input:focus {
        border-color: #FFD700 !important;
        box-shadow: 0 0 12px rgba(255, 215, 0, 0.4) !important;
    }
    .stTextArea textarea {
        background-color: #0d1440 !important;
        border: 1px solid #FFD700 !important;
        border-radius: 8px !important;
        color: #f5f5f5 !important;
        padding: 12px !important;
    }
    .stTextArea textarea:focus {
        border-color: #FFD700 !important;
        box-shadow: 0 0 12px rgba(255, 215, 0, 0.4) !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #FFD700 !important;
        color: #0a0a2e !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        background-color: #FFC200 !important;
        transform: scale(1.03) !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.4) !important;
    }

    /* Result Animation */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Result Box — Positive */
    .result-positive {
        background: linear-gradient(135deg, #0d2b0d, #0a1f0a);
        border: 1px solid #00cc44;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1.5rem;
        text-align: center;
        animation: fadeUp 0.6s ease forwards;
        box-shadow: 0 0 20px rgba(0, 204, 68, 0.2);
    }
    .result-positive h2 {
        color: #00cc44;
        font-size: 2rem;
        margin-bottom: 0.8rem;
    }

    /* Result Box — Negative */
    .result-negative {
        background: linear-gradient(135deg, #2b0d0d, #1f0a0a);
        border: 1px solid #cc0000;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1.5rem;
        text-align: center;
        animation: fadeUp 0.6s ease forwards;
        box-shadow: 0 0 20px rgba(204, 0, 0, 0.2);
    }
    .result-negative h2 {
        color: #cc0000;
        font-size: 2rem;
        margin-bottom: 0.8rem;
    }

    /* Result Details */
    .result-detail {
        color: #FFD700;
        font-size: 1rem;
        margin: 0.4rem 0;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Progress Bar */
    .stProgress > div > div {
        background-color: #FFD700 !important;
        border-radius: 999px !important;
    }
    .stProgress > div {
        background-color: #0d1440 !important;
        border-radius: 999px !important;
    }

    /* Stat Cards */
    .stat-card {
        background: #0d1440;
        border: 1px solid #1a2a6c;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-card h3 {
        color: #FFD700;
        font-size: 1.4rem;
        margin: 0;
    }
    .stat-card p {
        color: #7788cc;
        font-size: 0.75rem;
        margin: 0.2rem 0 0 0;
        letter-spacing: 1px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #334;
        font-size: 0.8rem;
        margin-top: 3rem;
        letter-spacing: 1px;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "show_about" not in st.session_state:
    st.session_state.show_about = False

if st.button("ℹ️ ABOUT"):
    st.session_state.show_about = not st.session_state.show_about

# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    with open("lr_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ============================================================
# Text Cleaning
# ============================================================
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("# 🎬 About")
    st.markdown("---")

    st.markdown("### 🤖 Model Info")
    st.markdown("""
    - **Model:** Logistic Regression
    - **Dataset:** IMDB Reviews
    - **Train Size:** 25,000
    - **Test Size:** 25,000
    """)

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3>87.69%</h3>
            <p>ACCURACY</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h3>0.8773</h3>
            <p>F1 SCORE</p>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h3>87.45%</h3>
            <p>PRECISION</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-card">
            <h3>88.01%</h3>
            <p>RECALL</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔧 Pipeline")
    st.markdown("""
    1. Raw Text Input
    2. Clean Text
    3. TF-IDF Vectorize
    4. LR Predict
    5. Result
    """)

    st.markdown("---")
    st.markdown("### 📚 Built With")
    st.markdown("""
    - Scikit-Learn
    - Hugging Face
    - Streamlit
    - Pandas & NumPy
    """)

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🎬 SENTIMENT ANALYSIS</h1>
    <p>IMDB MOVIE REVIEW CLASSIFIER</p>
</div>
<div class="gold-divider"></div>
""", unsafe_allow_html=True)

if st.session_state.show_about:
    st.info(
        """
        🎬 Sentiment Analysis App

        - Model: Logistic Regression
        - Dataset: IMDB Reviews
        - Built with Scikit-Learn and Streamlit
        - Uses a saved TF-IDF + Logistic Regression pipeline
        """
    )

# ============================================================
# Inputs
# ============================================================
movie_name = st.text_input(
    "🎥 MOVIE NAME",
    placeholder="e.g. The Dark Knight",
    key="movie_name"
)

review = st.text_area(
    "✍️ YOUR REVIEW",
    placeholder="Write your movie review here...",
    height=150,
    key="review"
)

# ============================================================
# Buttons
# ============================================================
col1, col2 = st.columns([3, 1])
with col1:
    analyze = st.button("🔍 ANALYZE SENTIMENT", use_container_width=True)
with col2:
    clear = st.button("🗑️ CLEAR", use_container_width=True)

# ============================================================
# Clear Logic
# ============================================================
if clear:
    for key in ["movie_name", "review"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ============================================================
# Analyze Logic
# ============================================================
if analyze:
    if not review.strip():
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("🎬 Analyzing..."):
            cleaned = clean_text(review)
            prediction = model.predict([cleaned])[0]
            confidence = model.predict_proba([cleaned])[0]
            conf_pct = round(max(confidence) * 100, 2)

            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <h2>✅ POSITIVE SENTIMENT</h2>
                    <p class="result-detail">Confidence : {conf_pct}%</p>
                    <p class="result-detail">Model Used : Logistic Regression</p>
                    {"<p class='result-detail'>🎬 Movie : " + movie_name + "</p>" if movie_name else ""}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h2>❌ NEGATIVE SENTIMENT</h2>
                    <p class="result-detail">Confidence : {conf_pct}%</p>
                    <p class="result-detail">Model Used : Logistic Regression</p>
                    {"<p class='result-detail'>🎬 Movie : " + movie_name + "</p>" if movie_name else ""}
                </div>
                """, unsafe_allow_html=True)

            st.progress(int(conf_pct))

# ============================================================
# Footer
# ============================================================
st.markdown("""
<div class="footer">
    BUILT WITH SCIKIT-LEARN + STREAMLIT | IMDB DATASET
</div>
""", unsafe_allow_html=True)
