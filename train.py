"""Run once: downloads the UCI data, tunes and compares models, saves models/model.joblib (takes a few minutes)."""
from src import core
B = core.train()
for n, s in sorted(B["cv"].items(), key=lambda x: -x[1]): print(f"  {n:22s} CV macro-F1 {s:.3f}")
print(f"Best: {B['best']} | test acc {B['acc']:.3f} | test macro-F1 {B['f1']:.3f} | no-grades baseline acc {B['acc_ng']:.3f}")
