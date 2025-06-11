from datetime import datetime, timedelta
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict
from tqdm import tqdm
from config import MISSING_VALUE

def cut_treatment_data_up_to_day_n(json_data, pid, adm_id, tr_id, day_n):
    cutoff = datetime.strptime(day_n, "%Y-%m-%d") + timedelta(days=1)
    pat = json_data[str(pid)]
    for adm in pat["admissions"]:
        if adm["PatientContactId"] != adm_id:
            continue
        for tr in adm["treatments"]:
            if tr["treatment_id"] != tr_id:
                continue
            tr_cut = deepcopy(tr)
            tr_cut["prescriptions"] = [
                p if pd.to_datetime(p["StopDatumTijd"]) < cutoff else {
                    **p,
                    "StopDatumTijd": (cutoff - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
                }
                for p in tr["prescriptions"] if pd.to_datetime(p["StartDatumTijd"]) < cutoff
            ]
            tr_cut["treatment_cultures"] = [
                c for c in tr.get("treatment_cultures", []) if pd.to_datetime(c["afnamedatumtijd"]) < cutoff
            ]
            tr_cut["treatment_end"] = (cutoff - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")

            adm_cut = deepcopy(adm)
            adm_cut["treatments"] = [tr_cut]
            for k in adm_cut.get("measurements", {}):
                adm_cut["measurements"][k] = [m for m in adm_cut["measurements"][k] if pd.to_datetime(m["MeetMoment"]) < cutoff]
            return adm_cut
    return None

def build_global_label_encoders(data: dict) -> dict:
    print("Building label encoders")
    values = {
        "med": set(),
        "route": set(),
        "specialisme": set(),
        "groep": set(),
        "setting": set(),
        "doel": set(),
        "locatie": set(),
        "info": set(),
        "materiaal": set(),
        "uitslag": set(),
        "genus": set(),
        "microbe_cat": set()
    }

    for pid, pat in data.items():
        for adm in pat["admissions"]:
            for tr in adm["treatments"]:
                values["locatie"].add(tr.get("locatie", MISSING_VALUE))
                values["doel"].add(tr.get("doel", MISSING_VALUE))
                for p in tr.get("prescriptions", []):
                    values["med"].add(p.get("MedicatieStofnaam", MISSING_VALUE))
                    values["route"].add(p.get("ToedieningsRoute", MISSING_VALUE))
                    values["specialisme"].add(p.get("Specialisme", MISSING_VALUE))
                    values["groep"].add(p.get("Specialisme_groep", MISSING_VALUE))
                    values["setting"].add(p.get("community_or_hospital", MISSING_VALUE))
                    values["doel"].add(p.get("doel", MISSING_VALUE))
                    values["locatie"].add(p.get("locatie", MISSING_VALUE))
                    values["info"].add(p.get("aanvullende_informatie", MISSING_VALUE))
                for c in tr.get("treatment_cultures", []):
                    values["materiaal"].add(c.get("materiaal_catCustom", MISSING_VALUE))
                    values["uitslag"].add(c.get("kweek_uitslagDef", MISSING_VALUE))
                    values["genus"].add(str(c.get("microbe_genus")) if c.get("microbe_genus") else MISSING_VALUE)
                    values["microbe_cat"].add(c.get("microbe_catCustom", MISSING_VALUE))

    encoders = {}
    for key, val_set in values.items():
        val_set.add(MISSING_VALUE)  # ensure fallback is encoded
        le = LabelEncoder()
        le.fit(list(val_set))
        encoders[key] = le
    return encoders