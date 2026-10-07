import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from src import core

LOW, AVG, HIGH, INK = "#D9482B", "#E9A23B", "#1F8A70", "#1F2350"
COL = {"Low": LOW, "Average": AVG, "High": HIGH}
st.set_page_config(page_title="Student Performance Classifier", page_icon="🎓", layout="wide")
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=Source+Sans+3:wght@400;600&display=swap');
html,body,[class*="css"],.stApp{{font-family:'Source Sans 3',sans-serif;background:#F3F5FA;color:{INK}}}
h1,h2,h3{{font-family:'Bricolage Grotesque',sans-serif!important;color:{INK};letter-spacing:-.02em}}
.hero h1{{font-size:2.6rem;font-weight:800;margin:0}} .hero p{{color:#5B6086;margin:.2rem 0 1rem;max-width:60ch}}
.card{{background:#fff;border-radius:14px;padding:1.2rem 1.4rem;border:1px solid #DDE1F0}}
.big{{font-family:'Bricolage Grotesque';font-weight:800;font-size:3.4rem;line-height:1}}
.pbar{{display:flex;height:16px;border-radius:8px;overflow:hidden;margin:.8rem 0 .4rem}}
.badge{{display:inline-block;padding:.15rem .7rem;border-radius:99px;color:#fff;font-weight:600;font-size:.9rem}}
.tip{{border-left:4px solid {AVG};background:#FFF8EA;padding:.55rem .8rem;margin:.4rem 0;border-radius:0 8px 8px 0}}
.stTabs [data-baseweb=tab]{{font-family:'Bricolage Grotesque';font-weight:700;font-size:1.05rem}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    if not core.MODEL.exists():
        with st.spinner("First run: downloading the UCI data and training the model..."): core.train()
    return core.load_bundle()


B = get_bundle(); D, O = B["defaults"], B["options"]
KEY = ["studytime", "failures", "absences", "higher", "subject", "age", "goout", "Dalc", "Walc", "health", "famrel", "freetime", "internet", "schoolsup"]
PRESETS = {"Struggling": dict(studytime=1, failures=2, absences=14, higher="no", goout=5, Dalc=3, Walc=5, health=2, schoolsup="no"),
           "Typical": dict(D),
           "Strong": dict(studytime=4, failures=0, absences=1, higher="yes", goout=2, Dalc=1, Walc=1, health=5, famrel=5)}


def apply(name):
    for k, v in PRESETS[name].items():
        if k in core.NUM: v = max(O[k][0], min(O[k][1], v))
        st.session_state[k] = v


def field(f, col):
    lab = core.LABELS[f]; st.session_state.setdefault(f, D[f])
    if f == "studytime": return col.select_slider(lab, [1, 2, 3, 4], key=f, format_func=core.STUDY.get)
    if f in core.CAT: return col.selectbox(lab, O[f], key=f)
    lo, hi = O[f]; return col.slider(lab, lo, hi, key=f)


def prob_bar(r):
    segs = "".join(f'<div style="width:{r[f"P({c})"]*100:.1f}%;background:{COL[c]}" title="{c}"></div>' for c in core.CLASSES)
    legend = " &nbsp; ".join(f'<span style="color:{COL[c]};font-weight:600">{c} {r[f"P({c})"]:.0%}</span>' for c in core.CLASSES)
    return f'<div class="pbar">{segs}</div><div>{legend}</div>'


def factor_chart(ex, n=8):
    t = ex.head(n).iloc[::-1]
    lbl = [f"{core.LABELS[f]}: {core.STUDY.get(v, v) if f == 'studytime' else v}" for f, v in zip(t.feature, t.value)]
    fig = go.Figure(go.Bar(x=t.impact, y=lbl, orientation="h", marker_color=[HIGH if i >= 0 else LOW for i in t.impact],
                           text=[f"{i:+.0%}" for i in t.impact], textposition="outside"))
    fig.update_layout(height=380, margin=dict(l=0, r=30, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="Effect on predicted class probability", zeroline=True, gridcolor="#E5E8F4"))
    return fig


st.markdown('<div class="hero"><h1>Where is this student heading?</h1><p>Enter a student\'s profile and get a High, Average or Low prediction with the reasons behind it, '
            'early-warning flags and concrete next steps. Trained on the UCI Student Performance data.</p></div>', unsafe_allow_html=True)
t1, t2, t3 = st.tabs(["Predict", "Batch (CSV)", "Model report"])

with t1:
    st.caption("Start from a sample student, then adjust anything. The result updates as you change values.")
    pc = st.columns(3 + 3)
    for c, n in zip(pc, PRESETS): c.button(f"Load: {n}", on_click=apply, args=(n,))
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        cols = st.columns(2)
        for i, f in enumerate(KEY): field(f, cols[i % 2])
        with st.expander("More background details"):
            c2 = st.columns(2)
            for i, f in enumerate([x for x in core.FEATURES if x not in KEY]): field(f, c2[i % 2])
    vals = {f: st.session_state.get(f, D[f]) for f in core.FEATURES}
    row = pd.DataFrame([vals]); r = core.predict(B, row).iloc[0]; ex = core.explain(B, row)
    rc = {"At risk": LOW, "Watch": AVG, "On track": HIGH}[r.Risk]
    with right:
        st.markdown(f'<div class="card"><span class="badge" style="background:{rc}">{r.Risk}</span>'
                    f'<div class="big" style="color:{COL[r.Prediction]};margin-top:.5rem">{r.Prediction}</div>'
                    f'<div style="color:#5B6086">Confidence {r.Confidence:.0%}</div>{prob_bar(r)}</div>', unsafe_allow_html=True)
        st.markdown("##### What drove this prediction")
        st.plotly_chart(factor_chart(ex))
        st.caption("Green pushes toward the predicted class, red pulls away from it. Measured against a typical student.")
    tips = core.advice(row, ex)
    if tips:
        st.markdown("##### Where to improve")
        for t in tips: st.markdown(f'<div class="tip">{t}</div>', unsafe_allow_html=True)
    with st.expander("What if this student changed one habit?"):
        w = st.columns(4)
        new = dict(vals)
        new["studytime"] = w[0].select_slider("Study time", [1, 2, 3, 4], value=vals["studytime"], format_func=core.STUDY.get, key=f"w1{vals['studytime']}")
        new["absences"] = w[1].slider("Absences", 0, 40, min(40, vals["absences"]), key=f"w2{vals['absences']}")
        new["failures"] = w[2].slider("Past failures", 0, 3, vals["failures"], key=f"w3{vals['failures']}")
        new["Walc"] = w[3].slider("Weekend alcohol", 1, 5, vals["Walc"], key=f"w4{vals['Walc']}")
        r2 = core.predict(B, pd.DataFrame([new])).iloc[0]
        d = r2["P(High)"] - r["P(High)"]; e = r2["P(Low)"] - r["P(Low)"]
        a, b, c = st.columns(3)
        a.metric("New prediction", r2.Prediction, f"was {r.Prediction}", delta_color="off")
        b.metric("Chance of High", f"{r2['P(High)']:.0%}", f"{d:+.0%}")
        c.metric("Chance of Low", f"{r2['P(Low)']:.0%}", f"{e:+.0%}", delta_color="inverse")

with t2:
    st.markdown("Upload a CSV with the UCI student columns. Missing columns are filled with typical values.")
    st.download_button("Download CSV template", pd.DataFrame([D]).to_csv(index=False), "student_template.csv")
    up = st.file_uploader("Student CSV", type="csv")
    if up:
        try:
            raw = pd.read_csv(up, sep=None, engine="python"); res = core.predict(B, raw)
            out = pd.concat([raw.reset_index(drop=True), res], axis=1)
            k = st.columns(4)
            k[0].metric("Students", len(out))
            for c, n in zip(k[1:], ["At risk", "Watch", "On track"]): c.metric(n, int((out.Risk == n).sum()))
            f = st.radio("Show", ["All", "At risk", "Watch", "On track"], horizontal=True)
            view = out if f == "All" else out[out.Risk == f]
            front = ["Prediction", "Confidence", "Risk"]
            st.dataframe(view[front + [c for c in view.columns if c not in front]].style.format({"Confidence": "{:.0%}"}), height=320)
            st.download_button("Download results", out.to_csv(index=False), "predictions.csv")
            st.plotly_chart(px.histogram(out, x="Prediction", color="Prediction", color_discrete_map=COL, category_orders={"Prediction": core.CLASSES}).update_layout(showlegend=False, height=260, margin=dict(t=10)))
            i = st.selectbox("Inspect a student", out.index, format_func=lambda i: f"Row {i}: {out.Prediction[i]} ({out.Confidence[i]:.0%})")
            ex_i = core.explain(B, raw.iloc[[i]]); st.plotly_chart(factor_chart(ex_i))
            for t in core.advice(core._clean(B, raw.iloc[[i]]), ex_i): st.markdown(f'<div class="tip">{t}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not read this file: {e}. Use the template above for the expected columns.")

with t3:
    m = st.columns(4)
    m[0].metric("Chosen model", B["best"]); m[1].metric("Test accuracy", f"{B['acc']:.1%}")
    m[2].metric("Test macro-F1", f"{B['f1']:.2f}"); m[3].metric("Students", B["n"])
    a, b = st.columns(2)
    with a:
        st.markdown("##### Confusion matrix (held-out 20%)")
        st.plotly_chart(px.imshow(B["cm"], x=core.CLASSES, y=core.CLASSES, text_auto=True, color_continuous_scale=["#EEF0FA", INK],
                                  labels=dict(x="Predicted", y="Actual", color="Students")).update_layout(height=360, margin=dict(t=10)))
    with b:
        st.markdown("##### Overall feature importance")
        imp = pd.Series(B["importance"]).nlargest(10).iloc[::-1]
        st.plotly_chart(go.Figure(go.Bar(x=imp.values, y=[core.LABELS[i] for i in imp.index], orientation="h", marker_color=INK)).update_layout(height=360, margin=dict(t=10, l=0)))
    st.markdown("##### Model comparison (5-fold CV, macro-F1)")
    st.dataframe(pd.Series(B["cv"], name="CV macro-F1").sort_values(ascending=False).to_frame().style.format("{:.3f}"))
    st.caption("Data: UCI Student Performance (Cortez & Silva, 2008), Math + Portuguese. Label from final grade G3: Low 0-9, Average 10-14, High 15-20. "
               "G1 and G2 are excluded because they are earlier grades that would leak the answer.")
