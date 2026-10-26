"""Speak Ateso: a Streamlit app that checks a learner's Ateso translation.

Run on your computer:   streamlit run app.py
"""
import random

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

import scorer

st.set_page_config(page_title="Speak Ateso: answer checker", page_icon="🗣️", layout="centered")

COLORS = {"Correct": "success", "Nearly right": "warning", "Wrong": "error"}


# ---------- 1. Load data and train the model (runs once, then cached) ----------
@st.cache_data
def load_questions():
    return pd.read_csv("questions.csv", keep_default_na=False)


@st.cache_resource(show_spinner="Training the scoring model (first start only)...")
def load_model():
    # Training here, instead of loading a saved model file, means the model always
    # matches the scikit-learn version installed on the server. It takes a few seconds.
    data = pd.read_csv("train_features.csv.gz")
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=42)
    model.fit(data[scorer.FEATURE_NAMES], data["label"])
    return model


def check_answer(model, prompt, reference, learner):
    feats = pd.DataFrame([scorer.features(prompt, reference, learner)])[scorer.FEATURE_NAMES]
    label = model.predict(feats)[0]
    confidence = float(model.predict_proba(feats).max())
    error = "none" if label == "Correct" else scorer.diagnose(prompt, reference, learner)
    if label != "Correct" and error == "none":
        error = "multiple_errors"
    return {"label": label, "confidence": confidence, "error": error,
            "feedback": scorer.FEEDBACK[error], "similarity": feats["chrf"].iloc[0]}


questions = load_questions()
model = load_model()


# ---------- 2. Remember the learner's progress between clicks ----------
def new_session():
    order = list(range(len(questions)))
    random.shuffle(order)
    st.session_state.update(order=order, pos=0, score=0, answered=0, streak=0, result=None)


if "order" not in st.session_state:
    new_session()

s = st.session_state
q = questions.iloc[s.order[s.pos % len(questions)]]


# ---------- 3. Sidebar: about and reset ----------
with st.sidebar:
    st.header("About")
    st.write("Type the Ateso translation of each English sentence. "
             "The checker says whether it is **Correct**, **Nearly right** or **Wrong**, "
             "and what kind of mistake it found.")
    st.caption("Sentences: SALT corpus by Sunbird AI and Makerere AI Lab (CC BY-SA 4.0). "
               "Scores come from a model trained on generated mistakes; real answers "
               "that are worded differently from the reference may be marked wrong.")
    if st.button("Start again"):
        new_session()
        st.rerun()


# ---------- 4. Main screen ----------
st.title("Speak Ateso")
st.caption("Practice translating into Ateso and get instant feedback.")

c1, c2, c3 = st.columns(3)
c1.metric("Score", f"{s.score} / {s.answered}")
c2.metric("Streak", s.streak)
c3.metric("Question", s.pos + 1)

st.subheader("Translate into Ateso:")
st.markdown(f"### “{q['english']}”")

with st.form(key=f"answer_{s.pos}"):
    answer = st.text_input("Your answer", placeholder="Type in Ateso, then press Enter")
    submitted = st.form_submit_button("Check", type="primary", disabled=s.result is not None)

if submitted:
    if not answer.strip():
        st.info("Type an answer first.")
    else:
        r = check_answer(model, q["english"], q["ateso"], answer)
        s.result = {**r, "answer": answer}
        s.answered += 1
        if r["label"] == "Correct":
            s.score += 1
            s.streak += 1
        else:
            s.streak = 0
        st.rerun()

if s.result:
    r = s.result
    getattr(st, COLORS[r["label"]])(f"**{r['label']}**: {r['feedback']}")
    st.markdown(f"**Correct answer:** {q['ateso']}")
    with st.expander("How was this judged?"):
        st.write(f"Your answer: {r['answer']}")
        st.write(f"Model confidence: {r['confidence']:.0%}")
        st.write(f"Mistake type: `{r['error']}`")
        st.write(f"Similarity to the reference (chrF): {r['similarity']:.0f} / 100")
    if st.button("Next question ➜", type="primary"):
        s.pos += 1
        s.result = None
        st.rerun()
else:
    if st.button("Skip / show answer"):
        s.result = {"label": "Wrong", "feedback": "Skipped.", "confidence": 1.0,
                    "error": "skipped", "similarity": 0.0, "answer": "(skipped)"}
        s.answered += 1
        s.streak = 0
        st.rerun()
