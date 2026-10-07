"""Data loading, training, prediction, explanation and advice for the Student Performance Classifier."""
import io, zipfile, urllib.request
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA, MODEL = ROOT / "data", ROOT / "models" / "model.joblib"
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
CLASSES = ["Low", "Average", "High"]          # G3 0-9 / 10-14 / 15-20
NUM = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
       "freetime", "goout", "Dalc", "Walc", "health", "absences"]
CAT = ["subject", "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
       "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
FEATURES = NUM + CAT   # G1/G2 deliberately excluded: they are earlier grades that leak G3.
LABELS = {"age": "Age", "Medu": "Mother's education (0-4)", "Fedu": "Father's education (0-4)",
          "traveltime": "Travel time to school (1-4)", "studytime": "Weekly study time", "failures": "Past class failures",
          "famrel": "Family relationship quality (1-5)", "freetime": "Free time after school (1-5)",
          "goout": "Going out with friends (1-5)", "Dalc": "Weekday alcohol use (1-5)", "Walc": "Weekend alcohol use (1-5)",
          "health": "Health (1-5)", "absences": "Absences", "subject": "Subject", "school": "School", "sex": "Sex",
          "address": "Home area", "famsize": "Family size", "Pstatus": "Parents' status", "Mjob": "Mother's job",
          "Fjob": "Father's job", "reason": "Reason for choosing school", "guardian": "Guardian",
          "schoolsup": "Extra school support", "famsup": "Family support", "paid": "Paid extra classes",
          "activities": "Extra-curriculars", "nursery": "Attended nursery", "higher": "Wants higher education",
          "internet": "Internet at home", "romantic": "In a relationship"}
STUDY = {1: "Under 2 h", 2: "2-5 h", 3: "5-10 h", 4: "Over 10 h"}

# (feature, trigger, advice) - used for "weaker factor" improvement suggestions
RULES = [("studytime", lambda v: v <= 2, "Study time is low. Aim for 5+ hours a week in short, regular sessions."),
         ("failures", lambda v: v >= 1, "Past failures are weighing on the result. Revisit the failed topics with a tutor or peer group."),
         ("absences", lambda v: v >= 8, "Absences are high. Each missed class compounds, so set an attendance target and talk to the teacher about catch-up."),
         ("higher", lambda v: v == "no", "Set a concrete goal (course or career) to build motivation for higher studies."),
         ("Dalc", lambda v: v >= 3, "Weekday alcohol use is high. Cutting it down protects study time and focus."),
         ("Walc", lambda v: v >= 4, "Weekend alcohol use is high. Keep weekends for rest so Monday starts fresh."),
         ("goout", lambda v: v >= 4, "Social time is crowding out study. Block fixed study slots before going out."),
         ("health", lambda v: v <= 2, "Health is low. Sleep and routine have a direct effect on grades, so consider a check-up."),
         ("famrel", lambda v: v <= 2, "Family relations are strained. A mentor or counsellor can offer support."),
         ("internet", lambda v: v == "no", "No home internet. Use the library or computer lab for online resources."),
         ("schoolsup", lambda v: v == "no", "Ask about extra school support or remedial sessions.")]


def _fetch():
    DATA.mkdir(exist_ok=True)
    def unpack(raw):
        z = zipfile.ZipFile(io.BytesIO(raw))
        for n in z.namelist():
            if n.endswith(".zip"): unpack(z.read(n))
            elif n.endswith(".csv"): (DATA / Path(n).name).write_bytes(z.read(n))
    unpack(urllib.request.urlopen(URL, timeout=60).read())


def load_data():
    if not (DATA / "student-mat.csv").exists(): _fetch()
    parts = []
    for f, s in [("student-mat.csv", "Math"), ("student-por.csv", "Portuguese")]:
        d = pd.read_csv(DATA / f, sep=";"); d["subject"] = s; parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["performance"] = pd.cut(d["G3"], [-1, 9, 14, 20], labels=CLASSES).astype(str)
    return d


def train():
    d = load_data(); X, y = d[FEATURES], d["performance"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pre = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    models = {"Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
              "Random Forest": RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
              "Gradient Boosting": GradientBoostingClassifier(random_state=42)}
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = {n: float(cross_val_score(Pipeline([("pre", pre), ("m", m)]), Xtr, ytr, cv=cv, scoring="f1_macro").mean()) for n, m in models.items()}
    best = max(scores, key=scores.get)
    pipe = Pipeline([("pre", pre), ("m", models[best])]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    imp = permutation_importance(pipe, Xte, yte, scoring="f1_macro", n_repeats=5, random_state=42)
    B = {"pipe": pipe, "best": best, "cv": scores, "n": len(d), "dist": y.value_counts().to_dict(),
         "acc": float(accuracy_score(yte, pred)), "f1": float(f1_score(yte, pred, average="macro")),
         "cm": confusion_matrix(yte, pred, labels=CLASSES), "importance": dict(zip(FEATURES, imp.importances_mean)),
         "defaults": {**{f: int(round(X[f].median())) for f in NUM}, **{f: X[f].mode()[0] for f in CAT}},
         "options": {**{f: (int(X[f].min()), int(X[f].max())) for f in NUM}, **{f: sorted(X[f].unique()) for f in CAT}}}
    MODEL.parent.mkdir(exist_ok=True); joblib.dump(B, MODEL)
    return B


def load_bundle(): return joblib.load(MODEL)


def _clean(B, df):
    X = df.reindex(columns=FEATURES).copy()
    X[NUM] = X[NUM].apply(pd.to_numeric, errors="coerce")
    return X.fillna(pd.Series(B["defaults"])).reset_index(drop=True)


def risk_level(pred, p_low):
    return "At risk" if pred == "Low" or p_low >= 0.5 else "Watch" if p_low >= 0.25 else "On track"


def predict(B, df):
    X = _clean(B, df); pipe = B["pipe"]
    P = pd.DataFrame(pipe.predict_proba(X), columns=pipe.classes_)[CLASSES]
    out = pd.DataFrame({"Prediction": P.idxmax(axis=1), "Confidence": P.max(axis=1)})
    for c in CLASSES: out[f"P({c})"] = P[c]
    out["Risk"] = [risk_level(a, b) for a, b in zip(out.Prediction, P["Low"])]
    return out


def explain(B, df):
    """Occlusion: how much does the predicted class's probability drop if a feature is reset to the typical value?"""
    X = _clean(B, df).iloc[[0]]; pipe = B["pipe"]
    p = pipe.predict_proba(X)[0]; k = p.argmax()
    M = pd.concat([X] * len(FEATURES), ignore_index=True)
    for i, f in enumerate(FEATURES): M.loc[i, f] = B["defaults"][f]
    q = pipe.predict_proba(M)[:, k]
    e = pd.DataFrame({"feature": FEATURES, "value": [X.iloc[0][f] for f in FEATURES], "impact": p[k] - q})
    return e.reindex(e.impact.abs().sort_values(ascending=False).index).reset_index(drop=True)


def advice(row, ex=None, limit=4):
    row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
    rank = {f: i for i, f in enumerate(ex.feature)} if ex is not None else {}
    hits = [(rank.get(f, 99), t) for f, test, t in RULES if f in row and test(row[f])]
    return [t for _, t in sorted(hits)][:limit]
