"""
src/models/train_churn.py
-------------------------
• 6 моделей: logreg, RF, GBT, LGBM, CatBoost, MLP
• два препроцессора:
      pre_ohe   – числовые + One-Hot (для logreg/RF/GBT/MLP)
      pre_native– числовые + raw-категории (для LGBM/CatBoost)
• функция tune_lgbm(n_trials) – Optuna-тюнинг
"""

# ────────────────── импорты & тихий режим ─────────────────────
from pathlib import Path
import warnings, numpy as np, pandas as pd, optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.features.feature_engineering import BalanceSalaryRatio, add_age_square

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names"
)

# ────────────────── 0. данные ─────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
df   = pd.read_csv(ROOT / "data" / "raw" / "churn" / "Churn_Modelling.csv")
y    = df.pop("Exited").values
df   = add_age_square(df)

num_cols = ["CreditScore", "Balance", "EstimatedSalary", "Age", "Age_Sq"]
cat_cols = ["Geography", "Gender"]
cat_idx  = [df.columns.get_loc(c) for c in cat_cols]

pre_ohe = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pre_native = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", "passthrough", cat_cols)          # необработанные категории
])

# ────────────────── 1. модели ─────────────────────────────────
# ── модели ────────────────────────────────────────────────────
models = {
    #       (ключ препроцессора,           модель)
    "logreg": ("ohe", LogisticRegression(max_iter=1000, class_weight="balanced")),
    "rf":     ("ohe", RandomForestClassifier(
                       n_estimators=700, max_depth=10,
                       class_weight="balanced", random_state=42)),
    "gbt":    ("ohe", GradientBoostingClassifier(subsample=0.8, learning_rate=0.05)),
    "mlp":    ("ohe", MLPClassifier(hidden_layer_sizes=(128, 64),
                                    alpha=1e-4, max_iter=600, random_state=42)),

    #  **LightGBM и CatBoost получают тоже OHE!**
    "lgbm":   ("ohe", LGBMClassifier(
                        n_estimators=700, max_depth=7, learning_rate=0.05,
                        scale_pos_weight=6371/1629, verbose=-1)),
    "cat":    ("ohe", CatBoostClassifier(
                        iterations=700, depth=6, learning_rate=0.05,
                        auto_class_weights="Balanced",
                        random_state=42, verbose=False))
}

# ────────────────── 2. функция оценки ─────────────────────────
def evaluate(pre_key: str, model):
    pre = pre_ohe if pre_key == "ohe" else pre_native
    cv  = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = []

    for tr, te in cv.split(df, y):
        pipe = Pipeline([
            ("fe", BalanceSalaryRatio()),
            ("pre", pre),
            ("model", model)
        ])
        pipe.fit(df.iloc[tr], y[tr])
        prob = pipe.predict_proba(df.iloc[te])[:, 1]
        scores.append(average_precision_score(y[te], prob))

    return np.mean(scores)

# ────────────────── 3. основной проход ────────────────────────
def main():
    for name, (key, mdl) in models.items():
        print(f"{name:6s}", round(evaluate(key, mdl), 5))

# ────────────────── 4. Optuna-тюнинг LGBM ─────────────────────
def tune_lgbm(n_trials: int = 40):
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("leaves", 20, 200),
            "max_depth":  trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("lr", 5e-3, 0.15, log=True),
            "n_estimators": trial.suggest_int("n", 300, 1200),
            "subsample": trial.suggest_float("sub", 0.6, 1.0),
            "scale_pos_weight": 6371 / 1629,
            "verbose": -1               # ← оставляем
            # "categorical_feature": cat_idx   ← УБРАТЬ!
        }
        pipe = Pipeline([
            ("fe", BalanceSalaryRatio()),
            ("pre", pre_ohe),            # ← здесь было pre_native
            ("model", LGBMClassifier(**params))
        ])
        cv, scores = StratifiedKFold(5, shuffle=True, random_state=42), []
        for tr, te in cv.split(df, y):
            pipe.fit(df.iloc[tr], y[tr])
            scores.append(average_precision_score(
                y[te], pipe.predict_proba(df.iloc[te])[:, 1]))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("◎ tuned LGBM PR-AUC:", round(study.best_value, 5))
    print("best params:", study.best_params)

# ────────────────── CLI ───────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tune_lgbm":
        tune_lgbm()
    else:
        main()
