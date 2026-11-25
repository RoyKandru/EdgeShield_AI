"""
Train a simple SMS baseline from a local CSV.

Expected CSV columns: label,text
Labels may be ham/spam or benign/scam. The script is intentionally separate from
the online rule-based starter detector. Dataset download and licensing must be
handled by the project team.
"""
import argparse
import csv
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="models/sms_baseline.joblib")
    args = parser.parse_args()

    texts, labels = [], []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get("text", "").strip()
            label = row.get("label", "").strip().lower()
            if text and label:
                texts.append(text)
                labels.append(label)

    if len(set(labels)) < 2 or len(texts) < 20:
        raise SystemExit("Need at least 20 rows and two distinct labels.")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(x_train, y_train)
    print(classification_report(y_test, model.predict(x_test), zero_division=0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Saved model: {args.out}")


if __name__ == "__main__":
    main()
