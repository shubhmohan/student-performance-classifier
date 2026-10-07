import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from src import core

BLUE, CYAN, GREEN, RED, ORANGE = "#0075FF", "#21D4FD", "#01B574", "#E31A1A", "#FFB547"
COL = {"Low": RED, "Average": ORANGE, "High": GREEN}
RISKC = {"At risk": RED, "Watch": ORANGE, "On track": GREEN}
st.set_page_config(page_title="Student Performance Classifier", page_icon="🎓", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;700;800&display=swap');
html,body,.stApp,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif}
.stApp{background:radial-gradient(900px 500px at 85% -5%,rgba(0,117,255,.45),rgba(0,117,255,0) 65%),radial-gradient(700px 400px at 0% 100%,rgba(33,212,253,.12),rgba(0,0,0,0) 60%),linear-gradient(160deg,#0B1437 0%,#060B28 55%,#050818 100%);color:#fff}
header[data-testid=stHeader]{background:transparent} #MainMenu,footer{visibility:hidden}
.block-container{padding-top:1.5rem;max-width:1400px}
section[data-testid=stSidebar]{background:linear-gradient(180deg,#0A1235,#060B28);border-right:1px solid rgba(255,255,255,.07)}
[data-testid=stVerticalBlockBorderWrapper]{background:linear-gradient(127deg,rgba(6,11,40,.94) 19%,rgba(10,14,35,.55) 76%);border:1px solid rgba(255,255,255,.09);border-radius:20px;box-shadow:0 10px 40px rgba(0,0,0,.25)}
h1,h2,h3,h4,label,p,span{color:#fff} small,[data-testid=stCaptionContainer],.stCaption{color:#A0AEC0!important}
.logo{font-weight:800;letter-spacing:.2em;font-size:.85rem;padding:.4rem .2rem} .sep{border:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent);margin:.8rem 0 1rem}
section[data-testid=stSidebar] [role=radiogroup]{gap:.35rem} section[data-testid=stSidebar] [role=radiogroup] label{padding:.65rem .9rem;border-radius:15px;width:100%;font-weight:700;font-size:.85rem;cursor:pointer}
section[data-testid=stSidebar] [role=radiogroup] label>div:first-child{display:none}
section[data-testid=stSidebar] [role=radiogroup] label:has(input:checked){background:#0F1535;box-shadow:0 7px 20px rgba(0,0,0,.35)}
.help{margin-top:2rem;padding:1.1rem;border-radius:20px;background:radial-gradient(circle at 20% 0,#0075FF,#4B2BC8 60%,#0B1437);font-size:.8rem}
.crumb{color:#A0AEC0;font-size:.75rem} .pt{font-size:1.15rem;font-weight:700;margin-bottom:1rem}
.hero{border-radius:20px;padding:1.6rem 1.8rem;margin-bottom:1rem;border:1px solid rgba(255,255,255,.1);background:radial-gradient(500px 220px at 90% 30%,rgba(0,117,255,.55),rgba(0,0,0,0) 70%),linear-gradient(127deg,rgba(6,11,40,.94),rgba(10,14,35,.5))}
.hero .hs{color:#A0AEC0;font-size:.85rem} .hero .ht{font-size:2.1rem;font-weight:800;margin:.15rem 0 .3rem;letter-spacing:-.02em}
.kpi{display:flex;justify-content:space-between;align-items:center;padding:1rem 1.2rem;border-radius:20px;border:1px solid rgba(255,255,255,.09);background:linear-gradient(127deg,rgba(6,11,40,.94),rgba(10,14,35,.55));margin-bottom:1rem}
.kl{color:#A0AEC0;font-size:.72rem;font-weight:700} .kv{font-size:1.25rem;font-weight:800} .ks{font-size:.75rem;font-weight:700;margin-left:.3rem}
.ic{width:42px;height:42px;border-radius:12px;background:#0075FF;display:flex;align-items:center;justify-content:center;font-size:1.1rem;box-shadow:0 0 18px rgba(0,117,255,.55)}
.ct{font-weight:700;font-size:1rem} .cs{color:#A0AEC0;font-size:.78rem;margin-bottom:.6rem}
.pr{margin:.7rem 0} .prl{display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:.25rem} .trk{height:6px;border-radius:4px;background:rgba(255,255,255,.1)} .trk div{height:6px;border-radius:4px}
.tip{display:flex;gap:.7rem;align-items:flex-start;margin:.65rem 0;font-size:.85rem} .dot{min-width:10px;height:10px;border-radius:50%;margin-top:.4rem;background:#FFB547;box-shadow:0 0 10px #FFB547}
div[data-baseweb=select]>div,div[data-baseweb=input]>div{background:#0F1535!important;border-color:rgba(226,232,240,.18)!important;border-radius:12px}
.stButton>button,.stDownloadButton>button{background:#0F1535;border:1px solid rgba(255,255,255,.15);border-radius:12px;color:#fff;font-weight:700}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#0075FF;box-shadow:0 0 16px rgba(0,117,255,.5);color:#fff}
[data-testid=stMetric]{background:rgba(255,255,255,.03);border-radius:14px;padding:.7rem 1rem}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    if not core.MODEL.exists():
        with st.spinner("First run: downloading the UCI data and training the model (a few minutes)..."): core.train()
    return core.load_bundle()


B = get_bundle(); D, O = B["defaults"], B["options"]
KEY = ["G1", "G2", "studytime", "failures", "absences", "higher", "subject", "age", "goout", "Dalc", "Walc", "health", "famrel", "freetime", "internet", "schoolsup"]
PRESETS = {"Struggling": dict(G1=7, G2=6, studytime=1, failures=2, absences=14, higher="no", goout=5, Dalc=3, Walc=5, health=2, schoolsup="no"),
           "Typical": dict(D),
           "Strong": dict(G1=16, G2=17, studytime=4, failures=0, absences=1, higher="yes", goout=2, Dalc=1, Walc=1, health=5, famrel=5)}


def apply(name):
    for k, v in PRESETS[name].items():
        if k in core.NUM: v = max(O[k][0], min(O[k][1], v))
        st.session_state[k] = v


def field(f, col):
    lab = core.LABELS[f]; st.session_state.setdefault(f, D[f])
    if f == "studytime": return col.select_slider(lab, [1, 2, 3, 4], key=f, format_func=core.STUDY.get)
    if f in core.CAT: return col.selectbox(lab, O[f], key=f)
    lo, hi = O[f]; return col.slider(lab, lo, hi, key=f)


def kpi(label, value, sub="", icon="📊", color="#01B574"):
    return f'<div class="kpi"><div><div class="kl">{label}</div><div class="kv">{value}<span class="ks" style="color:{color}">{sub}</span></div></div><div class="ic">{icon}</div></div>'


def title(t, s=""): st.markdown(f'<div class="ct">{t}</div><div class="cs">{s}</div>', unsafe_allow_html=True)


def ring(c, color):
    L = 251.3
    return (f'<svg viewBox="0 0 180 112" width="100%" style="max-width:320px;display:block;margin:auto"><defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="{BLUE}"/><stop offset="1" stop-color="{color}"/></linearGradient></defs>'
            f'<path d="M10 95 A80 80 0 0 1 170 95" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="12" stroke-linecap="round"/>'
            f'<path d="M10 95 A80 80 0 0 1 170 95" fill="none" stroke="url(#g)" stroke-width="12" stroke-linecap="round" stroke-dasharray="{L*c:.1f} {L}"/>'
            f'<text x="90" y="80" text-anchor="middle" fill="#fff" font-size="30" font-weight="800">{c:.0%}</text><text x="90" y="100" text-anchor="middle" fill="#A0AEC0" font-size="9">model confidence</text></svg>')


def probrows(r):
    return "".join(f'<div class="pr"><div class="prl"><span>{c}</span><b>{r[f"P({c})"]:.0%}</b></div><div class="trk"><div style="width:{r[f"P({c})"]*100:.1f}%;background:{COL[c]};box-shadow:0 0 10px {COL[c]}"></div></div></div>' for c in core.CLASSES)


def style(fig, h=340):
    fig.update_layout(height=h, margin=dict(l=0, r=10, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#fff", size=12, family="Plus Jakarta Sans"), xaxis=dict(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.25)"), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig


def factor_chart(ex, n=8):
    t = ex.head(n).iloc[::-1]
    lbl = [f"{core.LABELS[f]}: {core.STUDY.get(v, v) if f == 'studytime' else v}" for f, v in zip(t.feature, t.value)]
    return style(go.Figure(go.Bar(x=t.impact, y=lbl, orientation="h", marker_color=[GREEN if i >= 0 else RED for i in t.impact],
                                  text=[f"{i:+.0%}" for i in t.impact], textposition="outside", cliponaxis=False)), 360)


def tips_html(tips):
    return "".join(f'<div class="tip"><div class="dot"></div><div>{t}</div></div>' for t in tips) or '<div class="cs">No weak factors found. This profile looks healthy. 🎉</div>'


with st.sidebar:
    st.markdown('<div class="logo">🎓 &nbsp;STUDENT AI</div><div class="sep"></div>', unsafe_allow_html=True)
    page = st.radio("nav", ["🏠  Dashboard", "📂  Batch analysis", "📈  Model report"], label_visibility="collapsed")
    st.markdown('<div class="help"><b>Need help?</b><br>Start from a sample student, then change any value. Green factors support the prediction, red ones work against it.</div>', unsafe_allow_html=True)
name = page.split("  ")[1]
st.markdown(f'<div class="crumb">Pages / {name}</div><div class="pt">{name}</div>', unsafe_allow_html=True)

if name == "Dashboard":
    for f in core.FEATURES: st.session_state.setdefault(f, D[f])
    vals = {f: st.session_state[f] for f in core.FEATURES}
    row = pd.DataFrame([vals]); r = core.predict(B, row).iloc[0]; ex = core.explain(B, row)
    st.markdown('<div class="hero"><div class="hs">Student performance classifier</div><div class="ht">Where is this student heading?</div><div class="hs">Adjust the profile below. The prediction, reasons and advice update instantly.</div></div>', unsafe_allow_html=True)
    k = st.columns(4)
    k[0].markdown(kpi("Predicted performance", r.Prediction, f"  {r.Confidence:.0%}", "🎯", COL[r.Prediction]), unsafe_allow_html=True)
    k[1].markdown(kpi("Risk status", r.Risk, "", "🛡️", RISKC[r.Risk]), unsafe_allow_html=True)
    k[2].markdown(kpi("Chance of High", f"{r['P(High)']:.0%}", "", "📈"), unsafe_allow_html=True)
    k[3].markdown(kpi("Chance of Low", f"{r['P(Low)']:.0%}", "", "⚠️"), unsafe_allow_html=True)
    left, right = st.columns([1.55, 1], gap="medium")
    with left, st.container(border=True):
        title("Student profile", "Load a sample, then fine-tune")
        pc = st.columns(3)
        for c, n in zip(pc, PRESETS): c.button(n, on_click=apply, args=(n,), key=f"p_{n}")
        cols = st.columns(3)
        for i, f in enumerate(KEY): field(f, cols[i % 3])
        with st.expander("More background details"):
            c2 = st.columns(3)
            for i, f in enumerate([x for x in core.FEATURES if x not in KEY]): field(f, c2[i % 3])
    with right, st.container(border=True):
        title("Prediction confidence", "How sure the model is")
        st.markdown(ring(r.Confidence, COL[r.Prediction]) + probrows(r), unsafe_allow_html=True)
    a, b = st.columns([1.55, 1], gap="medium")
    with a, st.container(border=True):
        title("What drove this prediction", "Green supports the predicted class, red pulls away. Compared with a typical student.")
        st.plotly_chart(factor_chart(ex))
    with b, st.container(border=True):
        title("Where to improve", "Suggestions from the weakest factors")
        st.markdown(tips_html(core.advice(row, ex)), unsafe_allow_html=True)
    with st.container(border=True):
        title("What if this student changed one habit?", "Move a slider to re-run the prediction")
        w = st.columns(4); new = dict(vals)
        new["studytime"] = w[0].select_slider("Study time", [1, 2, 3, 4], value=vals["studytime"], format_func=core.STUDY.get, key=f"w1{vals['studytime']}")
        new["absences"] = w[1].slider("Absences", 0, 40, min(40, vals["absences"]), key=f"w2{vals['absences']}")
        new["failures"] = w[2].slider("Past failures", 0, 3, min(3, vals["failures"]), key=f"w3{vals['failures']}")
        new["Walc"] = w[3].slider("Weekend alcohol", 1, 5, vals["Walc"], key=f"w4{vals['Walc']}")
        r2 = core.predict(B, pd.DataFrame([new])).iloc[0]
        m = st.columns(3)
        m[0].metric("New prediction", r2.Prediction, f"was {r.Prediction}", delta_color="off")
        m[1].metric("Chance of High", f"{r2['P(High)']:.0%}", f"{r2['P(High)'] - r['P(High)']:+.0%}")
        m[2].metric("Chance of Low", f"{r2['P(Low)']:.0%}", f"{r2['P(Low)'] - r['P(Low)']:+.0%}", delta_color="inverse")

elif name == "Batch analysis":
    with st.container(border=True):
        title("Upload students", "CSV with the UCI student columns. Missing columns are filled with typical values.")
        st.download_button("Download CSV template", pd.DataFrame([D]).to_csv(index=False), "student_template.csv")
        up = st.file_uploader("Student CSV", type="csv", label_visibility="collapsed")
    if up:
        try:
            raw = pd.read_csv(up, sep=None, engine="python"); out = pd.concat([raw.reset_index(drop=True), core.predict(B, raw)], axis=1)
            k = st.columns(4)
            for c, (lab, v, ic) in zip(k, [("Students", len(out), "👥"), ("At risk", (out.Risk == "At risk").sum(), "🚨"), ("Watch", (out.Risk == "Watch").sum(), "👀"), ("On track", (out.Risk == "On track").sum(), "✅")]):
                c.markdown(kpi(lab, int(v), "", ic), unsafe_allow_html=True)
            with st.container(border=True):
                title("Results", "Filter by risk, then download")
                f = st.radio("Show", ["All", "At risk", "Watch", "On track"], horizontal=True, label_visibility="collapsed")
                view = out if f == "All" else out[out.Risk == f]; front = ["Prediction", "Confidence", "Risk"]
                st.dataframe(view[front + [c for c in view.columns if c not in front]].style.format({"Confidence": "{:.0%}"}), height=320)
                st.download_button("Download results", out.to_csv(index=False), "predictions.csv")
            a, b = st.columns([1, 1.4], gap="medium")
            with a, st.container(border=True):
                title("Class distribution")
                st.plotly_chart(style(px.histogram(out, x="Prediction", color="Prediction", color_discrete_map=COL, category_orders={"Prediction": core.CLASSES}).update_layout(showlegend=False), 330))
            with b, st.container(border=True):
                title("Inspect one student")
                i = st.selectbox("Student", out.index, format_func=lambda i: f"Row {i}: {out.Prediction[i]} ({out.Confidence[i]:.0%})", label_visibility="collapsed")
                ex_i = core.explain(B, raw.iloc[[i]]); st.plotly_chart(factor_chart(ex_i, 6))
                st.markdown(tips_html(core.advice(core._clean(B, raw.iloc[[i]]), ex_i)), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not read this file: {e}. Use the template above for the expected columns.")

else:
    k = st.columns(4)
    for c, (lab, v, s, ic) in zip(k, [("Chosen model", B["best"].split(" (")[0], "", "🤖"), ("Test accuracy", f"{B['acc']:.1%}", "", "🎯"), ("Test macro-F1", f"{B['f1']:.2f}", "", "📐"), ("Without term grades", f"{B['acc_ng']:.0%}", " accuracy", "📉")]):
        c.markdown(kpi(lab, v, s, ic, "#A0AEC0"), unsafe_allow_html=True)
    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        title("Confusion matrix", "Held-out 20% of students")
        fig = px.imshow(B["cm"], x=core.CLASSES, y=core.CLASSES, text_auto=True, color_continuous_scale=["#0B1437", BLUE], labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(style(fig.update_layout(coloraxis_showscale=False), 360))
    with b, st.container(border=True):
        title("Overall feature importance", "Permutation importance on macro-F1")
        imp = pd.Series(B["importance"]).nlargest(10).iloc[::-1]
        st.plotly_chart(style(go.Figure(go.Bar(x=imp.values, y=[core.LABELS[i] for i in imp.index], orientation="h", marker_color=CYAN)), 360))
    with st.container(border=True):
        title("Model comparison", "5-fold cross-validated macro-F1 on the training split")
        cv = pd.Series(B["cv"]).sort_values()
        st.plotly_chart(style(go.Figure(go.Bar(x=cv.values, y=cv.index, orientation="h", marker_color=BLUE, text=[f"{v:.3f}" for v in cv.values], textposition="outside", cliponaxis=False)).update_xaxes(range=[0, 1]), 260))
        st.caption("Data: UCI Student Performance (Cortez & Silva, 2008). Label from final grade G3: Low 0-9, Average 10-14, High 15-20. Term 1 and 2 grades are inputs because they are known before the final grade; "
                   f"without them accuracy falls to {B['acc_ng']:.0%}.")
