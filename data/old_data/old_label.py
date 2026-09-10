import pandas as pd
from pathlib import Path

# Пути (raw-строки или с прямыми слэшами)
OLD = Path(r"C:\dota_parser\parser\old_data")
NEW = Path(r"C:\dota_parser\parser\features_hero")

# Список новых файлов
new_files = list(NEW.glob("*_features.csv"))
print("→ Новые файлы:", [f.name for f in new_files])

for new_path in new_files:
    match_name = new_path.stem.replace("_features","")
    old_path   = OLD / f"{match_name}_features.csv"

    if not old_path.exists():
        print(f"⚠️  Старый файл не найден: {old_path.name}")
        continue

    # Читаем старый и новый
    df_old = pd.read_csv(old_path)
    df_new = pd.read_csv(new_path)

    # Проверка: одинаковое ли число строк?
    if len(df_old) != len(df_new):
        print(f"❌  Количество строк не совпадает: {old_path.name} ({len(df_old)}) vs {new_path.name} ({len(df_new)})")
        continue

    # Прямое копирование по индексам
    # Если в старом есть лишние колонки, выбираем именно label
    df_new['label'] = df_old['label'].astype(str).str.strip()

    # Сохраняем обратно
    df_new.to_csv(new_path, index=False)
    print(f"✅  Метки восстановлены в {new_path.name}")
