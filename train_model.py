import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# 1) Путь к папке с CSV-файлами
FOLDER = Path("data/old_data")

# 2) Загрузка и объединение
dfs = []
for file in sorted(FOLDER.glob("*.csv")):
    df = pd.read_csv(file, encoding="utf-8-sig")
    if 'label' in df.columns:
        dfs.append(df)
if not dfs:
    raise RuntimeError("Нет файлов с колонкой 'label' для обучения")
df_all = pd.concat(dfs, ignore_index=True)

# 3) Очистка меток
df_all['label'] = df_all['label'].astype(str).str.strip().str.lower()

# 4) Убираем метки roshan и split-push
df_all = df_all[~df_all['label'].isin(['roshan', 'split-push'])]

# 5) Распределение меток до отбора
print("📦 Распределение меток после удаления нежелательных классов:")
print(df_all['label'].value_counts(), "\n")

# 6) Признаки и целевая
X = df_all.drop(columns=['label'])
y = df_all['label']

# 7) Разбиение на train/test 85/15
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=42
)


# 9) Оценка важности признаков
tmp_rf = RandomForestClassifier(random_state=42, n_estimators=100)
tmp_rf.fit(X_train, y_train)
importances = tmp_rf.feature_importances_

# 10) Отбор по медиане
median_imp = np.median(importances)
low_idx = np.where(importances < median_imp)[0]
kept_idx = np.where(importances >= median_imp)[0]
print(f"❌ Отбрасываем {len(low_idx)} признаков (ниже медианы)\n")

# 11) Оставляем только высоковажные фичи и масштабируем их
X_train_sel = X_train.iloc[:, kept_idx].values
X_test_sel  = X_test.iloc[:, kept_idx].values

# 12) Оверсэмплинг меньшинств
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train_sel, y_train)

# 13) Автоматический подбор гиперпараметров
param_dist = {
    'n_estimators':     [50, 200, 400],
    'max_depth':        [None, 10, 20, 50],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight':     [None]
}
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1,
    random_state=42
)
search.fit(X_res, y_res)

print("🔍 Лучшие гиперпараметры:", search.best_params_, "\n")

# 14) Оценка на тестовой выборке
y_pred = search.best_estimator_.predict(X_test_sel)
print("=== Classification Report на тестовой выборке ===")
print(classification_report(y_test, y_pred, zero_division=0))

# 15) Сохранение артефактов
OUT = Path("models")
OUT.mkdir(exist_ok=True)
joblib.dump(search.best_estimator_, OUT / "model.joblib")
pd.Series(X.columns[kept_idx], name="feature") \
  .to_csv(OUT / "kept_features.csv", index=False)
print("\n Сохранено: models/model.joblib, models/kept_features.csv")
