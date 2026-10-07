# Student Performance Classifier

An ML app that classifies a student's performance as **High, Average or Low**, shows how confident the model is, explains which factors drove the result, flags students at risk, and suggests what to improve.

**Live demo:** _add your Streamlit Cloud link here_

## Features
- Prediction (High / Average / Low) with a confidence score and a probability bar for all three classes
- **Top contributing factors** for each student (per-student explanation, not just global importance)
- **Confusion matrix**, test accuracy, macro-F1 and a model comparison table (Model report tab)
- **At-risk flag** (At risk / Watch / On track) from the predicted class and P(Low)
- **Improvement suggestions** generated from a student's weaker factors, ordered by their impact
- **What-if simulator**: change study time, absences, failures or weekend alcohol use and see the prediction move
- **CSV upload** for many students, with risk filter, per-student inspection and a downloadable results file
- Dark glass dashboard UI (sidebar navigation, KPI cards, confidence gauge, factor chart)
- One-click sample students (Struggling / Typical / Strong) for quick demos

## Tech stack
Python, Pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer), Streamlit, Plotly, joblib.

## Setup
```bash
git clone https://github.com/shubhmohan/student-performance-classifier.git
cd student-performance-classifier
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py          # takes a few minutes: downloads the UCI data, compares models, saves models/model.joblib
streamlit run app.py
```
`train.py` downloads the dataset automatically. If your network blocks it, download the zip from the link below and put `student-mat.csv` and `student-por.csv` in `data/`.

## Dataset
[UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008): 1,044 students from two Portuguese schools (Math + Portuguese classes), 30+ attributes covering study habits, family background, school support, health and absences.
The target comes from the final grade **G3** (0-20): Low 0-9, Average 10-14, High 15-20.

## Approach
1. **Target:** G3 binned into three classes; the two subject files are merged, with `subject` kept as a feature.
2. **Prior grades:** term 1 and term 2 grades (G1, G2) are inputs because they are known before the final grade G3. The model is also evaluated without them as a baseline, which shows how much weaker habits and background are on their own.
3. **Feature engineering and preprocessing:** three engineered features inside the pipeline (average term grade, grade trend G2-G1, total alcohol use), then a `ColumnTransformer` with scaling for numeric columns and one-hot encoding for categorical ones, so the same steps run at prediction time.
4. **Model selection:** Logistic Regression, Random Forest and Gradient Boosting tuned with `RandomizedSearchCV` (stratified 5-fold, macro-F1), plus a soft-voting ensemble of the three. The best cross-validated model is kept. Macro-F1 is used because accuracy hides weak performance on the rarer classes.
5. **Evaluation:** stratified 80/20 hold-out, confusion matrix, accuracy and macro-F1.
6. **Explanations:** global importance via permutation importance. Per-student factors use occlusion: reset one feature to the typical value and measure the change in the predicted class probability.
7. **Risk and advice:** risk is *At risk* if the prediction is Low or P(Low) >= 50%, *Watch* if P(Low) >= 25%. Advice comes from interpretable rules triggered by weak values and ranked by factor impact.

## Challenges and how I solved them
- **Low accuracy without prior grades:** habits and background alone gave about 52% accuracy on 3 classes. I added the term 1 and 2 grades as inputs (they are known before the final), kept the no-grades model as a baseline, and report both.
- **Class imbalance:** High is the smallest class. I used stratified splits, class weights and macro-F1 for model selection.
- **Explaining a pipeline model:** one-hot encoding hides the original columns. I explained at the raw-feature level (occlusion and permutation importance) so the output is readable.
- **Noisy CSV uploads:** missing columns are filled with typical values and bad numbers are coerced, so partial files still work.
- **Limits:** the data is two schools in one country and the model predicts a grade band, not a person's potential. Treat output as a prompt for support, not a verdict.

## Project structure
```
app.py          Streamlit interface
train.py        Trains and saves the model
src/core.py     Data, training, prediction, explanation, advice
models/         Saved pipeline (created by train.py)
data/           UCI CSVs (downloaded by train.py)
```
