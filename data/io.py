
import random
import json
import pandas as pd
from labeling import _patient_has_iv_to_oral_switch


def load_json_subset(path: str, limit: int = 100, balanced: bool = True, seed: int = 42) -> dict:
    print("Loading data")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is None or not balanced:
        return dict(list(data.items())[:limit])

    random.seed(seed)
    pos, neg = [], []
    selected = {}

    for pid in random.sample(list(data.keys()), len(data)):
        pat = data[pid]
        if _patient_has_iv_to_oral_switch(pat):
            if len(pos) < limit // 2:
                pos.append(pid)
        else:
            if len(neg) < limit // 2:
                neg.append(pid)

        if len(pos) >= limit // 2 and len(neg) >= limit // 2:
            break

    selected_ids = pos + neg
    return {pid: data[pid] for pid in selected_ids}

def load_json_subset_only_iv(path: str, limit: int = 100, balanced: bool = True, seed: int = 42) -> dict:
    print("Loading data")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _first_prescription_is_iv(patient: dict) -> bool:
        first_presc = None
        for adm in patient.get("admissions", []):
            for tr in adm.get("treatments", []):
                for p in tr.get("prescriptions", []):
                    try:
                        start = pd.to_datetime(p["StartDatumTijd"])
                        route = str(p.get("ToedieningsRoute", "")).lower()
                        if first_presc is None or start < first_presc["start"]:
                            first_presc = {"start": start, "route": route}
                    except Exception as e:
                        print(f"[WARNING] Failed parsing prescription: {p} ({e})")
        return first_presc is not None and "intraveneus" in first_presc["route"]

    if limit is None or not balanced:
        return {pid: pat for pid, pat in data.items() if _first_prescription_is_iv(pat)}

    random.seed(seed)
    pos, neg = [], []

    for pid in random.sample(list(data.keys()), len(data)):
        pat = data[pid]
        if not _first_prescription_is_iv(pat):
            continue
        if _patient_has_iv_to_oral_switch(pat):
            if len(pos) < limit // 2:
                pos.append(pid)
        else:
            if len(neg) < limit // 2:
                neg.append(pid)
        if len(pos) >= limit // 2 and len(neg) >= limit // 2:
            break

    selected_ids = pos + neg
    return {pid: data[pid] for pid in selected_ids}