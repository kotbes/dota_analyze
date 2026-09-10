import os
import glob
import json
import numpy as np
import pandas as pd

INPUT_DIR  = os.path.join(os.getcwd(), "data", "outputs")
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "features_hero")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# радиус для плотности
DENSITY_RADIUS = 60

def process_file(jsonl_path, interval=30):
    print(f"\n👉 Обрабатываю: {os.path.basename(jsonl_path)}")
    # 1) Читаем все interval-записи
    recs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('type') == 'interval':
                recs.append(r)
    if not recs:
        print("   ⚠️ Нет записей interval, пропускаем")
        return None
    df = pd.DataFrame(recs)
    print(f"   Всего interval-записей: {len(df)}")

    # 2) Считаем бины и фильтруем по времени >= 0
    df = df[df['time'] >= 0].copy()
    df['bin'] = (df['time'] // interval).astype(int)
    bins = df['bin'].nunique()
    print(f"   Получилось {bins} бин(ов) по {interval} сек")

    # 3) Последнее состояние каждого героя в каждом бине
    df_last = (
        df.sort_values(['bin','slot','time'])
          .groupby(['bin','slot'], as_index=False)
          .last()
    )
    df_last = df_last[df_last['unit'].notna()]
    
    
    # Делаем pivot по slot:
    metrics = ['x','y','networth','kills','deaths']
    piv = df_last.pivot(
        index='bin',
        columns='slot',        # <- здесь было columns='hero'
        values=metrics
    ).fillna(0)

    # теперь MultiIndex: (metric, slot) превратим в slot_metric
    new_cols = []
    for metric, slot in piv.columns:
        new_cols.append(f"slot{slot}_{metric}")
    piv.columns = new_cols

    # 5) Агрегаты по бину
    agg = df_last.groupby('bin').agg({
        'x':        ['mean','std'],
        'y':        ['mean','std'],
        'networth': 'sum',
        'kills':    'sum',
        'deaths':   'sum',
        'roshans_killed':'sum',
        'towers_killed': 'sum',
        'obs_placed':'sum',
        'sen_placed':'sum'
    })
    agg.columns = [
        'avg_x','std_x','avg_y','std_y',
        'sum_networth','sum_kills','sum_deaths',
        'n_roshans','n_pushes','n_obs','n_sen'
    ]
    agg['n_wards'] = agg['n_obs'] + agg['n_sen']
    agg.drop(columns=['n_obs','n_sen'], inplace=True)

    # 6) Плотность героев: считаем для каждого бина
    def compute_density(gr):
        coords = gr[['x','y']].to_numpy()
        center = coords.mean(axis=0)
        dists  = np.linalg.norm(coords - center, axis=1)
        return (dists <= DENSITY_RADIUS).sum()

    density = (
        df_last
        .groupby('bin')
        .apply(compute_density)
        .rename('hero_density')
    )

    # 7) Networth Radiant vs Dire
    rad = df_last[df_last['slot']<5].groupby('bin')['networth'].sum()
    dire= df_last[df_last['slot']>=5].groupby('bin')['networth'].sum()
    diff = (rad - dire).rename('net_worth_diff')
    absdiff = diff.abs().rename('abs_net_worth_diff')
    sum_rad = rad.rename('sum_networth_radiant')
    sum_dir = dire.rename('sum_networth_dire')

    # 8) Собираем всё вместе
    full = pd.concat([
        piv,
        agg,
        density,
        sum_rad, sum_dir,
        diff, absdiff
    ], axis=1).fillna(0)

    # 9) Если где-то вдруг дубли колонок — убираем
    full = full.loc[:, ~full.columns.duplicated()]

    # 10) Сброс индекса bin → колонка + добавляем label
    full = full.reset_index().rename(columns={'index':'bin'})
    full['label'] = ''  # для ручной разметки

    print(f"   Итог: {full.shape[0]} строк × {full.shape[1]} признаков (включая bin,label)")
    return full

def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    print(f"Найдено файлов для обработки: {len(files)}")
    for p in files:
        df_feat = process_file(p)
        if df_feat is None:
            continue
        name = os.path.basename(p).replace(".jsonl", "_features.csv")
        out  = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(out):
            print(f"   ⏭️ Пропускаем: {out} — уже существует")
        else:
            df_feat.to_csv(out, index=False, encoding='utf-8-sig')
            print(f"   ✅ Сохранено: {out}")

    
    print("\n🎉 Готово! Все файлы сохранены в", OUTPUT_DIR)

if __name__ == "__main__":
    main()
