import json
from pathlib import Path
import pandas as pd

folder = Path("data/outputs")
rows = []

for file in folder.glob("*.jsonl"):
    seen_slots = set()
    
    # Извлекаем match_id
    if "_" in file.stem:
        match_id = file.stem.split("_")[1]
    else:
        match_id = file.stem

    with open(file, encoding="utf-8") as f:
        for line in f:
            if "CDOTA_Unit_Hero_" in line and '"slot":' in line:
                try:
                    data = json.loads(line)
                    unit = data.get("unit", "")
                    slot = data.get("slot", None)

                    # Один слот – одна запись
                    if unit.startswith("CDOTA_Unit_Hero_") and slot is not None and slot not in seen_slots:
                        hero_name = unit.replace("CDOTA_Unit_Hero_", "")
                        rows.append({
                            "match_id": match_id,
                            "slot": slot,
                            "hero_name": hero_name
                        })
                        seen_slots.add(slot)
                        if len(seen_slots) == 10:
                            break  # получили всех 10 — дальше не нужно
                except json.JSONDecodeError:
                    continue

df = pd.DataFrame(rows)
df.to_csv("data/hero_slots.csv", index=False, encoding="utf-8-sig")
print("✅ Сохранено hero_slots.csv — по одному герою на каждый слот.")
