"""Run once: downloads the UCI data, compares models, saves models/model.joblib."""
from src import core
B = core.train()
print(f"Best model: {B['best']} | CV macro-F1 {B['cv'][B['best']]:.3f} | test acc {B['acc']:.3f} | test macro-F1 {B['f1']:.3f}")
