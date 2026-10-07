"""Data loading, training, prediction, explanation and advice for the Student Performance Classifier."""
import io, zipfile, urllib.request
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA, MODEL = ROOT / "data", ROOT / "models" / "model.joblib"
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
CLASSES = ["Low", "Average", "High"]          # G3 0-9 / 10-14 / 15-20
NUM = ["G1", "G2", "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
       "freetime", "goout", "Dalc", "Walc", "health", "absences"]
CAT = ["subject", "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
       "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]
FEATURES = NUM + CAT   # G1/G2 = term 1/2 grades: known before the final grade G3, so valid inputs.
LABELS = {"G1": "Term 1 grade (0-20)", "G2": "Term 2 grade (0-20)", "age": "Age", "Medu": "Mother's education (0-4)", "Fedu": "Father's education (0-4)",
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
RULES = [("G2", lambda v: v < 10, "Last term's grade is below the pass mark. Book extra help on the weakest topics now, before the final."),
         ("G1", lambda v: v < 10, "The first term grade was weak. Review the basics of that term with a tutor or peers."),
         ("studytime", lambda v: v <= 2, "Study time is low. Aim for 5+ hours a week in short, regular sessions."),
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


ENG = ["grade_avg", "grade_trend", "alc_total"]   # engineered inside the pipeline


def _eng(X):
    X = X.copy()
    X["grade_avg"] = (X["G1"] + X["G2"]) / 2          # overall academic level
    X["grade_trend"] = X["G2"] - X["G1"]              # improving or slipping
    X["alc_total"] = X["Dalc"] + X["Walc"]
    return X


def _eng_ng(X):
    X = X.copy(); X["alc_total"] = X["Dalc"] + X["Walc"]; return X


def _make(model, num, fn):
    pre = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("eng", FunctionTransformer(fn)), ("pre", pre), ("m", model)])


def train():
    d = load_data(); X, y = d[FEATURES], d["performance"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    num, cv = NUM + ENG, StratifiedKFold(5, shuffle=True, random_state=42)

    def tune(model, grid, n_iter=12):
        s = RandomizedSearchCV(_make(model, num, _eng), {f"m__{k}": v for k, v in grid.items()}, n_iter=n_iter, cv=cv,
                               scoring="f1_macro", n_jobs=-1, random_state=42).fit(Xtr, ytr)
        return s.best_estimator_.named_steps["m"], float(s.best_score_)

    lr, s_lr = tune(LogisticRegression(max_iter=3000, class_weight="balanced"), dict(C=[0.03, 0.1, 0.3, 1, 3]), 5)
    rf, s_rf = tune(RandomForestClassifier(class_weight="balanced_subsample", random_state=42),
                    dict(n_estimators=[300, 500], max_depth=[None, 6, 10, 14], min_samples_leaf=[1, 2, 3], max_features=["sqrt", 0.4, 0.6]))
    gb, s_gb = tune(GradientBoostingClassifier(random_state=42),
                    dict(n_estimators=[100, 200, 300], learning_rate=[0.03, 0.05, 0.1], max_depth=[2, 3, 4], subsample=[0.7, 0.85, 1.0]))
    ens = VotingClassifier([("rf", clone(rf)), ("gb", clone(gb)), ("lr", clone(lr))], voting="soft")
    s_ens = float(cross_val_score(_make(ens, num, _eng), Xtr, ytr, cv=cv, scoring="f1_macro").mean())
    models = {"Logistic Regression": lr, "Random Forest": rf, "Gradient Boosting": gb, "Ensemble (soft vote)": ens}
    scores = {"Logistic Regression": s_lr, "Random Forest": s_rf, "Gradient Boosting": s_gb, "Ensemble (soft vote)": s_ens}
    best = max(scores, key=scores.get)
    pipe = _make(clone(models[best]), num, _eng).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    nb = [f for f in NUM if f not in ("G1", "G2")]      # baseline: no prior grades
    base = _make(clone(models[best]), nb + ["alc_total"], _eng_ng).fit(Xtr[nb + CAT], ytr)
    acc_ng = float(accuracy_score(yte, base.predict(Xte[nb + CAT])))
    imp = permutation_importance(pipe, Xte, yte, scoring="f1_macro", n_repeats=5, random_state=42)
    B = {"pipe": pipe, "best": best, "cv": scores, "n": len(d), "dist": y.value_counts().to_dict(),
         "acc": float(accuracy_score(yte, pred)), "acc_ng": acc_ng, "f1": float(f1_score(yte, pred, average="macro")),
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
