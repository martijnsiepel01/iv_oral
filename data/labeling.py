import pandas as pd
from datetime import timedelta
from typing import List

def _patient_has_iv_to_oral_switch(patient: dict, max_horizon: int = 3) -> bool:
    for adm in patient.get("admissions", []):
        for tr in adm.get("treatments", []):
            labels = label_iv_oral_switch_windows(tr, max_horizon)
            for row in labels:
                if any(row[f"switch_on_day_n+{h}"] for h in range(1, max_horizon + 1)):
                    return True
    return False

def label_iv_oral_switch_windows(treatment: dict, max_horizon: int = 3) -> List[dict]:
    daily = {}
    for p in treatment["prescriptions"]:
        s, e = pd.to_datetime(p["StartDatumTijd"]), pd.to_datetime(p["StopDatumTijd"])
        route_val = str(p.get("ToedieningsRoute", "")).lower()
        if route_val == "":
            print("[WARNING] Empty route in prescription:", p)
            print("Treatment context:", treatment)
        route = "iv" if "intraveneus" in route_val else "oraal"
        d = s.normalize()
        while d <= e.normalize():
            daily.setdefault(d, set()).add(route)
            d += timedelta(days=1)

    df = pd.DataFrame(
        {"date": list(daily.keys()), "route": ["iv" if "iv" in v else "oraal" for v in daily.values()]}
    )
    out = []
    for i, r in df.iterrows():
        if r.route != "iv":
            continue
        row = {"day_n": r.date.strftime("%Y-%m-%d")}
        for h in range(1, max_horizon + 1):
            j = i + h
            row[f"switch_on_day_n+{h}"] = int(df.iloc[j].route == "oraal") if j < len(df) else 0
        out.append(row)
    return out


def label_long_iv_windows(treatment: dict, horizon: int = 5) -> List[dict]:
    """Return list of windows indicating if IV continues for `horizon` days."""
    daily = {}
    for p in treatment["prescriptions"]:
        s, e = pd.to_datetime(p["StartDatumTijd"]), pd.to_datetime(p["StopDatumTijd"])
        route_val = str(p.get("ToedieningsRoute", "")).lower()
        route = "iv" if "intraveneus" in route_val else "oraal"
        d = s.normalize()
        while d <= e.normalize():
            daily.setdefault(d, []).append(route)
            d += timedelta(days=1)

    df = pd.DataFrame({"date": list(daily.keys()), "route": ["iv" if "iv" in v else "oraal" for v in daily.values()]})
    out = []
    for i, r in df.iterrows():
        if r.route != "iv":
            continue
        long_iv = True
        for h in range(1, horizon + 1):
            j = i + h
            if j >= len(df) or df.iloc[j].route != "iv":
                long_iv = False
                break
        out.append({"day_n": r.date.strftime("%Y-%m-%d"), "long_iv": int(long_iv)})
    return out

def earliest_switch_class(row: dict, binary: bool = False) -> int:
    if binary:
        return int(row["switch_on_day_n+1"])
    if row["switch_on_day_n+1"]:
        return 1
    if row["switch_on_day_n+2"]:
        return 2
    if row["switch_on_day_n+3"]:
        return 3
    return 0