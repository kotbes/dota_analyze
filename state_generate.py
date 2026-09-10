import pandas as pd
import numpy as np
import joblib
from sklearn.utils import shuffle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# === Папка с вашими фичами + колонками 'label' и 'win' ===
FOLDER = Path("data/winrate")

# 1) Загрузка и объединение всех CSV
dfs = []
for fp in sorted(FOLDER.glob("*.csv")):
    df = pd.read_csv(fp, encoding="utf-8-sig")
    if {'label','win'}.issubset(df.columns):
        # убираем лишние лейблы
        df = df[~df['label'].isin(['Roshan', 'def'])]
        dfs.append(df)
if not dfs:
    raise RuntimeError("Нет файлов с колонкой 'label' и 'win'")
df_all = pd.concat(dfs, ignore_index=True)

# 2) One-hot кодируем label
df_all = pd.get_dummies(df_all, columns=['label'])

# 3) Убираем из X не-числовые колонки
#    (time, bin, win)
to_drop = ['time', 'bin', 'win']
present = [c for c in to_drop if c in df_all.columns]
X = df_all.drop(columns=present)  # теперь в X только числовные и one-hot
y = df_all['win'].astype(int)

# 4) Честный split: 80% train, 20% test
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
#Баланс
# 4.5) Собираем DataFrame для удобства:
df_tr = X_tr.copy()
df_tr['win'] = y_tr.values

# 4.6) Определяем, какие ряды — Farm
#    колонка 'label_farm' — ваш one-hot признак
farm = df_tr[df_tr['label_farm'] == 1]
not_farm = df_tr[df_tr['label_farm'] == 0]

# 4.7) Оставляем все не-Farm, а из Farm берём, скажем, 20%:
farm_downsampled = farm.sample(frac=0.2, random_state=42)

# 4.8) Склеиваем обратно и перемешиваем:
df_tr_balanced = pd.concat([not_farm, farm_downsampled], ignore_index=True)
df_tr_balanced = shuffle(df_tr_balanced, random_state=42)

# 4.9) Разворачиваем обратно в X_tr_imp и y_tr_imp
y_tr_bal = df_tr_balanced['win'].astype(int)
X_tr_bal = df_tr_balanced.drop(columns=['win'])

# === 6) Подготавливаем sample_weight для *этого* X_tr_bal ===
label_gang = X_tr_bal['label_gang'].values
label_push = X_tr_bal['label_push'].values

sw = np.ones(len(y_tr_bal), dtype=float)
sw[label_gang == 1] *= 2.5
sw[label_push == 1] *= 2.5
# 5) Импутация медианой (только по числовым и one-hot!)
imputer = SimpleImputer(strategy='median')
X_tr_imp = imputer.fit_transform(X_tr_bal)
X_te_imp = imputer.transform(X_te)

# Сохранить импьютер
Path("models").mkdir(exist_ok=True)
joblib.dump(imputer, "models/win_imputer.joblib")

# === 6) Подготавливаем sample_weight для *этого* X_tr_bal ===
label_gang = X_tr_bal['label_gang'].values
label_push = X_tr_bal['label_push'].values

sw = np.ones(len(y_tr_bal), dtype=float)
sw[label_gang == 1] *= 2.0
sw[label_push == 1] *= 2.0

# 8) Обучение регрессора Q(s,a)
reg = RandomForestRegressor(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
reg.fit(X_tr_imp, y_tr_bal, sample_weight=sw)

# Сохранить модель
joblib.dump(reg, "models/win_regressor.joblib")

# 7) Оценка на hold-out
y_prob = reg.predict(X_te_imp)
y_pred = (y_prob >= 0.5).astype(int)

print("R² на тестовой выборке:", round(r2_score(y_te, y_prob), 3))
print("Accuracy:",           accuracy_score(y_te, y_pred))
print("ROC AUC:",            roc_auc_score(y_te, y_prob))
print("\nClassification Report:\n", classification_report(y_te, y_pred, zero_division=0))
print("Confusion matrix:\n",   confusion_matrix(y_te, y_pred))

# 8) Подготовка списков фичей из X.columns
feature_names = X.columns.tolist()
label_cols    = [c for c in feature_names if c.startswith("label_")]
numeric_cols  = [c for c in feature_names if not c.startswith("label_")]

# 9) Функция рекомендаций
def recommend(state: pd.Series):
    best_a, best_q = None, -np.inf
    for lbl in label_cols:
        # собираем один пример
        row = {col: state[col] for col in numeric_cols}
        for col in label_cols:
            row[col] = 1 if col == lbl else 0
        X0 = pd.DataFrame([row], columns=feature_names)
        X0_imp = imputer.transform(X0)
        q = reg.predict(X0_imp)[0]
        if q > best_q:
            best_q, best_a = q, lbl.replace("label_","")
    return best_a, best_q

# 10) Покажем пару рекомендаций на тесте
print("\nПримеры рекомендаций на hold-out:")
for idx in range(5):
    s = X_te.reset_index(drop=True).iloc[idx]
    action, qval = recommend(s)
    print(f"  #{idx+1}: текущее состояние → рекомендация: {action}, Q = {qval:.3f}")
