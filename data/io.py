
import random
import json
import pandas as pd
from data.labeling import _patient_has_iv_to_oral_switch


# ----------------------------------------------------------------------
# Shared balancing helper
# ----------------------------------------------------------------------
def _balance_dataset(patients: dict) -> dict:
    """Return a class-balanced copy of `patients` (0 = no switch, 1 = switch)."""
    pos = {pid: pat for pid, pat in patients.items()
           if _patient_has_iv_to_oral_switch(pat)}
    neg = {pid: pat for pid, pat in patients.items()
           if pid not in pos}
    n = min(len(pos), len(neg))
    return {**dict(list(pos.items())[:n]),
            **dict(list(neg.items())[:n])}


def load_json_subset(path: str,
                     limit: int | None = 100,
                     balanced: bool = True,
                     seed: int = 42) -> dict:
    print("Loading data")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is None and balanced:
        print("[INFO] Overriding 'balanced=True' in unlimited mode to match old behavior")
        balanced = False

    # ── 1. Unlimited load ────────────────────────────────────────────
    if limit is None or limit <= 0:        # treat None or ≤0 as “no limit”
        return data if not balanced else _balance_dataset(data)

    # ── 2. Unbalanced sample of size `limit` ─────────────────────────
    if not balanced:
        # take the first `limit` items (fast); itertools.islice saves memory
        from itertools import islice
        return dict(islice(data.items(), limit))

    # ── 3. Balanced sample (pos/neg split) ──────────────────────────
    random.seed(seed)
    pos, neg = [], []

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


def load_json_subset_only_iv(path: str,
                             limit: int | None = 100,
                             balanced: bool = True,
                             seed: int = 42) -> dict:
    """
    Load a subset of the JSON data **restricted to patients whose very first
    prescription is IV**. Supports three modes:

    1. Unlimited (`limit is None` or `limit <= 0`)
       • balanced=True  → return the whole IV cohort, balanced pos/neg  
       • balanced=False → return the whole IV cohort, unbalanced

    2. Finite `limit`, unbalanced (`balanced=False`)
       • first `limit` IV-first patients (deterministic order)

    3. Finite `limit`, balanced (`balanced=True`)
       • random sample: `limit//2` pos + `limit//2` neg
    """
    print("Loading data")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is None and balanced:
        print("[INFO] Overriding 'balanced=True' in unlimited mode to match old behavior")
        balanced = False

    # ------------------------------------------------------------
    # Helper: does this patient start with IV?
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 1. Unlimited load (limit None or ≤0)
    # ------------------------------------------------------------
    if limit is None or limit <= 0:
        iv_patients = {pid: pat for pid, pat in data.items()
                       if _first_prescription_is_iv(pat)}
        return iv_patients if not balanced else _balance_dataset(iv_patients)

    # ------------------------------------------------------------
    # 2. Finite limit, unbalanced
    # ------------------------------------------------------------
    if not balanced:
        from itertools import islice
        iv_iter = (item for item in data.items()
                   if _first_prescription_is_iv(item[1]))
        return dict(islice(iv_iter, limit))

    # ------------------------------------------------------------
    # 3. Finite limit, balanced
    # ------------------------------------------------------------
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
