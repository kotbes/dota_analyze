import re
from pathlib import Path
import pandas as pd

def main():
    # Жестко прописанные пути к папкам (при необходимости измените)
    features_dir = Path("data/features_hero")
    slots_dir    = Path("data/by_match")
    out_dir      = features_dir / "renamed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Жесткий порядок ролей (Radiant и Dire)
    role_order = [
        "core_r", "mid_r", "hard_r", "h_sup_r", "sup_r",
        "core_d",  "mid_d",  "hard_d",  "h_sup_d",  "sup_d"
    ]

    # Обрабатываем каждый файл признаков в директории
    for feat_path in features_dir.glob("*_features.csv"):
        name = feat_path.stem  # например "a_features" или "match_123456_features"
        
        # Извлекаем идентификатор матча из имени файла
        m = re.match(r"match_(.+?)_features", name)
        match_id = m.group(1) if m else name.replace("_features", "")

        # По этому ID ищем CSV со слотами и ролями
        slots_path = slots_dir / f"match_{match_id}_slots.csv"
        if not slots_path.exists():
            print(f"[WARN] не нашёл слотов для матча {match_id}, пропускаю.")
            continue

        # Читаем соответствие slot -> role
        df_slots = pd.read_csv(slots_path)
        slot2role = {int(s): r for s, r in zip(df_slots["slot"], df_slots["role"])}

        # Читаем файл признаков
        df = pd.read_csv(feat_path)

        # Переименовываем колонки вида slotX_<rest> в <role>_<rest>
        new_names = {}
        for col in df.columns:
            m2 = re.match(r"slot(\d+)_(.+)", col)
            if m2:
                slot = int(m2.group(1))
                rest = m2.group(2)
                role = slot2role.get(slot)
                if role is None:
                    raise KeyError(f"В {slots_path.name} нет слота {slot}")
                new_names[col] = f"{role}_{rest}"
        df = df.rename(columns=new_names)

        # Формируем итоговый порядок колонок:
        # 1) оригинальные колонки без префиксов ролей и без label
        non_slot = [c for c in df.columns if not any(c.startswith(r + "_") for r in role_order) and c != "label"]
        final_cols = non_slot.copy()

        # 2) колонки по ролям в заданном порядке
        for role in role_order:
            cols = [c for c in df.columns if c.startswith(role + "_")]
            final_cols.extend(cols)

        # 3) любые оставшиеся колонки, кроме label
        remaining = [c for c in df.columns if c not in final_cols and c != "label"]
        final_cols.extend(remaining)

        # 4) в конец добавляем label, если он есть
        if "label" in df.columns:
            final_cols.append("label")

        # Перестраиваем DataFrame и сохраняем
        df = df[final_cols]
        out_path = out_dir / feat_path.name
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] {feat_path.name} → {out_path.name}")

if __name__ == "__main__":
    main()
