import pandas as pd
from pathlib import Path

# папка с исходными *_features.csv
INPUT_DIR  = Path("data/winrate")
# папка, куда будем сохранять новые файлы с колонкой time
OUTPUT_DIR = Path("data/features_with_time")
OUTPUT_DIR.mkdir(exist_ok=True)

for src in sorted(INPUT_DIR.glob("*.csv")):
    df = pd.read_csv(src, encoding="utf-8-sig")

    if 'bin' not in df.columns:
        print(f"⚠️ пропускаю {src.name}: нет столбца bin")
        continue

    # добавляем колонку time — timedelta, конвертируем в строку HH:MM:SS
    df['time'] = pd.to_timedelta(df['bin'] * 30, unit='s')
    df['time'] = df['time'].dt.components[['hours','minutes','seconds']]\
                        .apply(lambda row: f"{int(row.hours):02d}:"
                                           f"{int(row.minutes):02d}:"
                                           f"{int(row.seconds):02d}", axis=1)

    # сохраняем результат
    dst = OUTPUT_DIR / src.name
    df.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"✅ {src.name} → {dst.name} (добавлен столбец time)")
