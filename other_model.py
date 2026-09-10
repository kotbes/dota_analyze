import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# 1) Загрузка
FOLDER = Path("data/features_hero")
dfs = []
for f in sorted(FOLDER.glob("*.csv")):
    df = pd.read_csv(f, encoding="utf-8-sig")
    if 'label' in df.columns:
        df = df[~df['label'].isin(['roshan','split-push'])]
        dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

# 2) X, y
X = df.drop(columns='label')
y = df['label'].str.strip().str.lower()

# 3) split 95/5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, stratify=y, random_state=42
)

# 4) Pipeline: impute → scale → SMOTE → RF(balanced_subsample)
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("smote",   SMOTE(random_state=42)),
    ("rf",      RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced_subsample",
                    random_state=42))
])

# 5) Быстрый CV
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
print("❓ Средний макро-F1 на CV (5-fold):", round(scores.mean(), 3))

# 6) Учим и отчёт на тесте
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print("\n=== Report на тестовой выборке ===")
print(classification_report(y_test, y_pred, zero_division=0))
