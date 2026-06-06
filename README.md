# 🎬 Sentiment Classifier — IMDB Movie Reviews

## Overview
An end-to-end Machine Learning Pipeline to classify IMDB movie reviews
as **Positive** or **Negative** sentiment using Scikit-Learn and XGBoost.

---

## 📊 Final Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression 🏆 | 0.8769 | 0.8745 | 0.8801 | 0.8773 |
| XGBoost | 0.8408 | 0.8203 | 0.8729 | 0.8458 |

**Winner: Logistic Regression** — outperformed XGBoost on all 4 metrics.

---

## 🔁 Pipeline Flow
Raw Text
↓
Text Cleaning (HTML removal, punctuation, lowercase)
↓
sklearn Pipeline (TF-IDF + Model)
↓
┌─────────────────────┐
↓                     ↓
Logistic Regression      XGBoost
↓                     ↓
Predictions          Predictions
└──────────┬──────────┘
↓
Evaluation
(Accuracy, Precision, Recall, F1)
↓
Visualization
(Confusion Matrix + Comparison Chart)
↓
Written Analysis

---

## 📁 Repository Structure
📁 sentiment-classifier
├── 📓 sentiment_classifier.ipynb  ← Main notebook (fully run)
├── 🖼️ confusion_matrix.png        ← Confusion matrix plot
├── 🖼️ model_comparison.png        ← Model comparison chart
└── 📄 README.md                   ← This file

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| Hugging Face Datasets | Load IMDB dataset |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-Learn | TF-IDF, Pipeline, LR, Metrics |
| XGBoost | XGBoost Classifier |
| Matplotlib | Plots and charts |
| Seaborn | Confusion matrix heatmap |
| Re | Text cleaning |

---

## 📋 Tasks Completed

- ✅ Load IMDB dataset from Hugging Face
- ✅ Exploratory Data Analysis (EDA)
- ✅ Text cleaning — HTML, punctuation, lowercase
- ✅ TF-IDF Vectorization inside sklearn Pipeline
- ✅ Train Logistic Regression Pipeline
- ✅ Train XGBoost Pipeline
- ✅ Evaluate — Accuracy, Precision, Recall, F1
- ✅ Confusion Matrix Plot (saved as image)
- ✅ Model Comparison Chart (saved as image)
- ✅ Written Analysis — Winner & Why
- ✅ Inference Pipeline — Real world input

---

## 🖼️ Visualizations

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Model Comparison Chart
![Model Comparison](model_comparison.png)

---

## 🔍 Key Findings

- Logistic Regression is best for **sparse TF-IDF text features**
- sklearn Pipeline ensures **clean, leakage-free, production-ready** code
- Perfectly balanced dataset ensured **fair model comparison**
- Inference pipeline works on **any new raw text input**

---

## 🚀 How to Run

```bash
# Step 1: Install libraries
pip install datasets scikit-learn xgboost 
    matplotlib seaborn pandas numpy

# Step 2: Open notebook
jupyter notebook sentiment_classifier.ipynb

# Step 3: Run All Cells
Kernel → Restart & Run All
```

---

## 📊 Dataset

- **Name:** IMDB Movie Reviews
- **Source:** Hugging Face Datasets
- **Size:** 50,000 reviews (25k train + 25k test)
- **Labels:** 0 = Negative | 1 = Positive
- **Balance:** Perfectly balanced (12,500 each)

---
