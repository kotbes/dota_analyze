import pandas as pd
from pathlib import Path

# 1) путь к исходному файлу
src = Path("data/hero_slots.csv")

# 2) папка для результатов
out_dir = Path("data/by_match")
out_dir.mkdir(exist_ok=True)

# 3) читаем мастер-файл
df = pd.read_csv(src)

# 4) для каждого match_id создаём свой CSV
for match_id, group in df.groupby("match_id"):
    out_path = out_dir / f"match_{match_id}_slots.csv"
    # сохраняем только нужные колонки (match_id, slot, hero_name, role)
    group.to_csv(out_path, index=False)
    print(f"→ {out_path} ({len(group)} записей)")

print("Разбиение завершено.")
