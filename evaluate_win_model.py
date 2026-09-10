import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# 1) Загрузка и конкатенация всех CSV с колонками 'label' и 'win'
FOLDER = Path("data/winrate")
dfs = []
for f in sorted(FOLDER.glob("*.csv")):
    df = pd.read_csv(f, encoding="utf-8-sig")
    if {'label','win'}.issubset(df.columns):
        # сразу отсеиваем ненужные стратегии
        df = df[~df['label'].isin(['roshan','split-push'])]
        dfs.append(df)
if not dfs:
    raise RuntimeError("Нет файлов с колонкой 'win' и 'label'")
df_all = pd.concat(dfs, ignore_index=True)

# 2) One-hot кодирование стратегий
df_all = pd.get_dummies(df_all, columns=['label'])

# 3) Отделяем X и y
y_true = df_all['win'].astype(int).values
X_full = df_all.drop(columns=['win']).values  # все фичи

# 4) Импутация пропусков по медиане
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X_full)

# 5) Загрузка обученного регрессора
reg = joblib.load("models/win_regressor.joblib")

# 6) Предсказания вероятностей
y_prob = reg.predict(X_imp)

#  если вдруг predict вернул не-числа, попробуем predict_proba
if not np.issubdtype(y_prob.dtype, np.number):
    if hasattr(reg, "predict_proba"):
        y_prob = reg.predict_proba(X_imp)[:,1]
    else:
        raise ValueError("Модель вернула не-числовые предсказания и не поддерживает predict_proba")

# 7) Бинаризация по 0.5 и метрики
y_pred = (y_prob >= 0.5).astype(int)

print("Accuracy:",   accuracy_score(y_true, y_pred))
print("ROC AUC:",    roc_auc_score(y_true, y_prob))
print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
