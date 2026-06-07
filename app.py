# ============================================================
# Sentiment Analysis App — Streamlit
# Model: Logistic Regression (IMDB Movie Reviews)
# ============================================================

import streamlit as st
import pickle
import re
from pathlib import Path

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎬",
    layout="centered"
)

# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "lr_pipeline.pkl"
    if not model_path.exists():
        st.error("Model file not found: lr_pipeline.pkl")
        return None

    try:
        with model_path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None

model = load_model()

# ============================================================
# Text Cleaning Function (Same as Training)
# ============================================================
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)         # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation
    text = text.lower().strip()               # lowercase
    return text

# ============================================================
# UI — Header
# ============================================================
st.markdown("# 🎬 Sentiment Analysis App")
st.markdown("**IMDB Movie Review Classifier**")
st.divider()

# ============================================================
# UI — Inputs
# ============================================================
movie_name = st.text_input(
    "Enter Movie Name:",
    placeholder="e.g. The Dark Knight"
)

review = st.text_area(
    "Enter Review:",
    placeholder="Write your movie review here...",
    height=150
)

# ============================================================
# UI — Analyze Button
# ============================================================
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if not review.strip():
        st.warning("Please enter a review first!")

    else:
        if model is None:
            st.error("The sentiment analysis model is unavailable. Please check the model file and restart the app.")
        else:
            with st.spinner("Analyzing..."):

                # Clean + Predict
                cleaned = clean_text(review)
                try:
                    prediction = model.predict([cleaned])[0]
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    prediction = None

                confidence = None
                if prediction is not None and hasattr(model, "predict_proba"):
                    try:
                        probability = model.predict_proba([cleaned])[0]
                        confidence = round(max(probability) * 100, 2)
                    except Exception:
                        confidence = None

                st.divider()

                if prediction is None:
                    st.error("Unable to compute sentiment for this review.")
                else:
                    # ── Result ──
                    if prediction == 1:
                        st.success("✅ Positive Sentiment")
                    else:
                        st.error("❌ Negative Sentiment")

                    if confidence is not None:
                        st.metric(
                            label="Confidence",
                            value=f"{confidence}%"
                        )
                        st.progress(min(max(int(confidence), 0), 100))

                    # ── Model Info ──
                    st.info("**Model Used:** Logistic Regression")

                    if movie_name:
                        st.caption(f"🎬 Movie: {movie_name}")

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("Built with Scikit-Learn + Streamlit | IMDB Dataset")