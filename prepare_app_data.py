"""Prepare the two data files the app needs, from the Week 2 attempts.

Run once (in Colab or on your computer), in the same folder as attempts.csv and scorer.py:
    python prepare_app_data.py

It writes:
    questions.csv            short test-set sentences the app asks the learner to translate
    train_features.csv.gz    the training features and labels; the app trains its model from these
"""
import pandas as pd
import scorer

att = pd.read_csv("attempts.csv", keep_default_na=False)

# 1. Question bank: short sentences from the TEST split only, so the app never quizzes
#    on sentences the model was trained on. 4-9 words keeps them manageable for learners.
q = att[(att["split"] == "test") & (att["label"] == "Correct")].copy()
q["n_words"] = q["reference"].str.split().str.len()
q = q[q["n_words"].between(4, 9)].drop_duplicates("reference")
q = q.sample(min(200, len(q)), random_state=42)
q[["prompt", "reference"]].rename(columns={"prompt": "english", "reference": "ateso"}) \
    .to_csv("questions.csv", index=False)
print(f"questions.csv: {len(q)} questions")

# 2. Training features: the same 16 numbers as Week 3, for the TRAIN split only.
tr = att[att["split"] == "train"]
X = pd.DataFrame([scorer.features(p, r, l) for p, r, l in
                  zip(tr["prompt"], tr["reference"], tr["learner_input"])])[scorer.FEATURE_NAMES]
X["label"] = tr["label"].values
X.round(4).to_csv("train_features.csv.gz", index=False)
print(f"train_features.csv.gz: {len(X):,} rows")
