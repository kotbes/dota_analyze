import pandas as pd
from pathlib import Path

# 1) Путь к папке с исходными CSV-файлами (запускать из корня проекта)
FEATURES_DIR = Path("data/features_hero")

# 2) Папка для сохранения файлов с колонкой win
OUT_DIR = Path("data/winrate")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 3) Запрос у пользователя: какие файлы/матчи Radiant выиграл?
winners_input = input(
    "Введите через запятую имена файлов (без .csv), в которых Radiant выиграл: "
)
winners = {name.strip() for name in winners_input.split(",") if name.strip()}

if not winners:
    print("⚠️ Ничего не введено, скрипт завершится.")
    exit(1)

# 4) Обработка каждого файла
for csv_path in sorted(FEATURES_DIR.glob("*.csv")):
    match_id = csv_path.stem  # имя файла без расширения
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Добавляем столбец win: 1 если match_id в списке winners, иначе 0
    df["win"] = 1 if match_id in winners else 0

    # Записываем в новую папку
    out_path = OUT_DIR / csv_path.name
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✔ Сохранён {out_path.name}: win={df['win'].iloc[0]} во всех строках")

print("\n✅ Все файлы сохранены в:", OUT_DIR)
