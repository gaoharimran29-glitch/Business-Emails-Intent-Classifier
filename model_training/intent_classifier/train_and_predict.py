"""
train_and_predict.py
Email Intent Classification using TF-IDF + Logistic Regression
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from dataset import get_dataframe

# ── 1. Load & Preprocess ────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Basic text cleaning: lowercase + strip."""
    return text.lower().strip()

def load_data():
    df = get_dataframe()
    df["clean_text"] = df["email_text"].apply(preprocess)
    return df

# ── 2. Train ─────────────────────────────────────────────────────────────────

def train(df):
    X = df["clean_text"]
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectoriser  (unigrams + bigrams, top 5000 features)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # Logistic Regression classifier
    model = LogisticRegression(max_iter=1000, random_state=42, C=5.0)
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    acc    = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print("  MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"\n  Test-set Accuracy : {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    return vectorizer, model

# ── 3. Save / Load ──────────────────────────────────────────────────────────

def save_model(vectorizer, model, path="model"):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{path}/intent_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Model saved to '{path}/' directory.")

def load_model(path="model"):
    with open(f"{path}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{path}/intent_model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

# ── 4. Predict ───────────────────────────────────────────────────────────────

def predict_intent(email_text: str, vectorizer, model) -> dict:
    """
    Returns predicted intent and confidence scores for a single email.
    """
    clean = preprocess(email_text)
    vec   = vectorizer.transform([clean])
    intent      = model.predict(vec)[0]
    proba       = model.predict_proba(vec)[0]
    classes     = model.classes_
    confidence  = dict(zip(classes, np.round(proba * 100, 2)))
    return {"intent": intent, "confidence": confidence}

# ── 5. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Train ---
    df = load_data()
    vectorizer, model = train(df)

    # --- Save model + dataset ---
    save_model(vectorizer, model)
    os.makedirs("data", exist_ok=True)
    df[["email_text", "intent"]].to_csv("data/email_intent_dataset.csv", index=False)
    print("  Dataset saved to 'data/email_intent_dataset.csv'")

    # ── 6. Example Predictions ───────────────────────────────────────────────
    test_emails = [
        "Can you send pricing details for enterprise plan?",
        "I would like a demo of your CRM platform next week.",
        "Your software keeps crashing every time I open it!",
        "How do I integrate your CRM with Zapier?",
        "We are not interested. Please remove us from your list.",
        "I want to see a demo and also know the pricing.",
        "What is the refund policy if we cancel early?",
        "The billing page is broken – I can't update my card.",
        "Can I get a free trial before committing to a plan?",
        "We have selected a different vendor. Thanks anyway.",
    ]

    print("\n" + "=" * 60)
    print("  EXAMPLE PREDICTIONS")
    print("=" * 60)
    for email in test_emails:
        result = predict_intent(email, vectorizer, model)
        top_conf = result["confidence"][result["intent"]]
        print(f"\n  Email   : {email}")
        print(f"  Intent  : {result['intent']}  (confidence: {top_conf}%)")
