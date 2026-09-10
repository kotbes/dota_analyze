import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from matplotlib import pyplot as plt

# === Настройки ===
FEATURES_DIR = Path("data/features_hero")
OUT_PNG      = Path.cwd() / "recommendations.png"
WARN_THRESH  = 0.5  # порог для предупреждений

# 1) Берём ВСЕ CSV в папке (или один файл — неважно)
files = sorted(FEATURES_DIR.glob("*.csv"))
if not files:
    raise RuntimeError("Папка пуста!")
print(f"Найдено файлов: {len(files)}")

# — для наглядности: если файлов >1, ты можешь выбирать конкретный через files[0]
df = pd.concat((pd.read_csv(f, encoding="utf-8-sig") for f in files),
               ignore_index=True)

# 2) Фильтруем лейблы и удаляем 'label'
df = df[~df['label'].isin(['Roshan'])]
df = df.drop(columns=['label'])

# 3) Переводим bin в секунды
if 'bin' not in df.columns:
    raise RuntimeError("Нет столбца bin!")
df['time_s'] = df['bin'].astype(int) * 30

# 4) Загружаем imputer и модель
imputer = joblib.load("models/win_imputer.joblib")
reg     = joblib.load("models/win_regressor.joblib")

# 5) Узнаём, какие признаки ждут imputer
all_feats    = list(imputer.feature_names_in_)
label_cols   = [c for c in all_feats if c.startswith("label_")]
numeric_cols = [c for c in all_feats if not c.startswith("label_")]

# 6) ДОзаполняем нулями все отсутствующие колонки (и выбрасываем лишние)
for col in numeric_cols + label_cols:
    if col not in df.columns:
        df[col] = 0
# теперь гарантированно df содержит ВСЕ нужные признаки

# 7) Собираем X_full и пропускаем через imputer
X_full = df[numeric_cols + label_cols]
X_imp  = imputer.transform(X_full)     # (N, P)

N, P = X_imp.shape
K    = len(label_cols)
print(f"P (число фич) = {P}, K (стратегий) = {K}")

# 8) Векторизуем один горячий разворот:
numeric_part = X_imp[:, :len(numeric_cols)]  # (N, numeric_count)
eyeK         = np.eye(K, dtype=int)          # (K, K)
block_oh     = np.tile(eyeK, (N, 1))         # (N*K, K)
num_rep      = np.repeat(numeric_part, K, axis=0)  # (N*K, numeric_count)
X_big        = np.hstack([num_rep, block_oh])     # (N*K, P)

# 9) Предсказываем Q и разворачиваем в (N, K)
Q_big = reg.predict(X_big).reshape(N, K)
team_idx = label_cols.index('label_team_fight')
push_idx = label_cols.index('label_push')
gang_idx = label_cols.index('label_gang')
Q_big[:, push_idx] += 0.0005
Q_big[:, gang_idx] += 0.001
Q_big[:, team_idx] += 0.001
# 10) Находим для каждой строки лучшую стратегию
best_idx   = Q_big.argmax(axis=1)
best_q     = Q_big[np.arange(N), best_idx]
best_label = [label_cols[i].replace("label_","") for i in best_idx]

# 11) Выводим рекомендации только при смене стратегии
prev = None
for t, lbl, q in zip(df['time_s'], best_label, best_q):
    if lbl != prev:
        m, s = divmod(int(t), 60)
        print(f"На {m:02d}:{s:02d} рекомендую начать {lbl.upper()} (Q = {q:.2f})")
        prev = lbl

# 12) Предупреждения о низком Q
for t, lbl, q, row in zip(df['time_s'], best_label, best_q, Q_big):
    if q < WARN_THRESH:
        tmp = row.copy()
        tmp[best_idx[np.where(best_q==q)[0][0]]] = -np.inf
        alt_j   = tmp.argmax()
        alt_lbl = label_cols[alt_j].replace("label_","")
        m, s    = divmod(int(t), 60)
        print(f"ВНИМАНИЕ {m:02d}:{s:02d}: стратегию {lbl.upper()} "
              f"следует сменить (Q={q:.2f}) → {alt_lbl.upper()} "
              f"(Q={row[alt_j]:.2f})")

# 13) Рисуем график Q(s,a) vs time
plt.figure(figsize=(10,6))
for j, lbl in enumerate(label_cols):
    plt.plot(df['time_s'], Q_big[:,j], label=lbl.replace("label_",""))
changes = np.where(np.array(best_label[:-1]) != np.array(best_label[1:]))[0] + 1
for idx in changes:
    t0 = df['time_s'].iat[idx]
    plt.axvline(t0, color="gray", linestyle="--", alpha=0.5)
    plt.text(t0, 1.02, best_label[idx].upper(), rotation=90, va="bottom", fontsize=8)

plt.xlabel("Time (s)")
plt.ylabel("Q(s,a)")
plt.title("Q-функции стратегий и моменты смен")
plt.legend(loc="upper right", bbox_to_anchor=(1.3,1))
plt.tight_layout()
plt.savefig(OUT_PNG)
print(f"\n🔍 График сохранён в {OUT_PNG}")
