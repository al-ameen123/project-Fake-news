"""
train_model.py — Train the Fake News Detector model.

This script loads the Kaggle "Fake and Real News Dataset", preprocesses the
text, trains a PassiveAggressiveClassifier with TF-IDF features, evaluates
performance, and saves the trained model + vectorizer to disk.

Usage:
    python train_model.py

Prerequisites:
    Place 'True.csv' and 'Fake.csv' from Kaggle into the 'data/' directory.
"""

import os
import sys
import time
import json

# Fix Windows console encoding for emoji/unicode output
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

from utils import preprocess_text


# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")
FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "pac_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MAX_ITER = 50


def main():
    print("=" * 60)
    print("  🗞️  FAKE NEWS DETECTOR — Model Training Pipeline")
    print("=" * 60)

    # ─── Step 1: Load Dataset ────────────────────────────────────────────

    print("\n📂 Step 1: Loading dataset...")

    if not os.path.isfile(TRUE_CSV):
        print(f"  ❌ ERROR: '{TRUE_CSV}' not found.")
        print("  → Download the dataset from Kaggle and place CSV files in 'data/'")
        print("  → https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        sys.exit(1)

    if not os.path.isfile(FAKE_CSV):
        print(f"  ❌ ERROR: '{FAKE_CSV}' not found.")
        print("  → Download the dataset from Kaggle and place CSV files in 'data/'")
        sys.exit(1)

    df_true = pd.read_csv(TRUE_CSV)
    df_fake = pd.read_csv(FAKE_CSV)

    print(f"  ✅ Real news articles loaded: {len(df_true):,}")
    print(f"  ✅ Fake news articles loaded: {len(df_fake):,}")

    # ─── Step 2: Label & Merge ───────────────────────────────────────────

    print("\n🏷️  Step 2: Labeling and merging datasets...")

    df_true["label"] = 1  # 1 = Real
    df_fake["label"] = 0  # 0 = Fake

    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Combine title + text for richer features
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"  ✅ Combined dataset: {len(df):,} articles")
    print(f"     Real: {(df['label'] == 1).sum():,}  |  Fake: {(df['label'] == 0).sum():,}")

    # ─── Step 3: Preprocess Text ─────────────────────────────────────────

    print("\n🧹 Step 3: Preprocessing text (this may take a minute)...")
    t_start = time.time()

    df["clean_text"] = df["content"].apply(preprocess_text)

    t_elapsed = time.time() - t_start
    print(f"  ✅ Preprocessing complete in {t_elapsed:.1f}s")

    # Drop any rows where cleaning produced empty text
    before_count = len(df)
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)
    dropped = before_count - len(df)
    if dropped > 0:
        print(f"  ⚠️  Dropped {dropped} empty rows after cleaning")

    # ─── Step 4: Train/Test Split ────────────────────────────────────────

    print(f"\n✂️  Step 4: Splitting data ({int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} train/test)...")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    print(f"  ✅ Training samples: {len(X_train):,}")
    print(f"  ✅ Testing samples:  {len(X_test):,}")

    # ─── Step 5: TF-IDF Vectorization ───────────────────────────────────

    print(f"\n📊 Step 5: Vectorizing text (TF-IDF, max_features={MAX_FEATURES}, ngrams={NGRAM_RANGE})...")
    t_start = time.time()

    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words="english",
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    t_elapsed = time.time() - t_start
    print(f"  ✅ Vectorization complete in {t_elapsed:.1f}s")
    print(f"     Feature matrix shape: {X_train_tfidf.shape}")

    # ─── Step 6: Train PassiveAggressiveClassifier ───────────────────────

    print(f"\n🧠 Step 6: Training PassiveAggressiveClassifier (max_iter={MAX_ITER})...")
    t_start = time.time()

    pac = PassiveAggressiveClassifier(max_iter=MAX_ITER, random_state=RANDOM_STATE)
    pac.fit(X_train_tfidf, y_train)

    t_elapsed = time.time() - t_start
    print(f"  ✅ Training complete in {t_elapsed:.1f}s")

    # ─── Step 7: Evaluate ────────────────────────────────────────────────

    print("\n📈 Step 7: Evaluating model performance...")

    y_pred = pac.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["FAKE", "REAL"])

    print(f"\n  🎯 ACCURACY: {accuracy * 100:.2f}%\n")
    print("  Confusion Matrix:")
    print(f"                  Predicted FAKE  Predicted REAL")
    print(f"    Actual FAKE     {cm[0][0]:>6,}          {cm[0][1]:>6,}")
    print(f"    Actual REAL     {cm[1][0]:>6,}          {cm[1][1]:>6,}")
    print(f"\n  Classification Report:\n")
    for line in report.split("\n"):
        print(f"    {line}")

    # ─── Step 8: Save Model ──────────────────────────────────────────────

    print("\n💾 Step 8: Saving model and vectorizer...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(pac, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)

    # Save metadata for the Streamlit app
    metadata = {
        "accuracy": round(accuracy * 100, 2),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "max_features": MAX_FEATURES,
        "ngram_range": list(NGRAM_RANGE),
        "confusion_matrix": cm.tolist(),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Model saved to:      {MODEL_PATH}")
    print(f"  ✅ Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"  ✅ Metadata saved to:   {METADATA_PATH}")

    print("\n" + "=" * 60)
    print("  ✅ Training complete! Run your app with: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
