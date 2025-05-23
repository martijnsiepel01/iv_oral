from datetime import datetime, timedelta
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict
from tqdm import tqdm

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

def build_global_label_encoders(data: dict) -> Dict[str, LabelEncoder]:
    print("Building label encoders")
    values_per_key = {
        "med": set(), "route": set(), "specialisme": set(), "groep": set(), "setting": set(),
        "doel": set(), "locatie": set(), "info": set(), "materiaal": set(), "uitslag": set(),
        "genus": set(), "microbe_cat": set()
    }
    for patient in tqdm(data.values(), desc="Iterating patients for global label encoders"):
        for admission in patient.get("admissions", []):
            for treatment in admission.get("treatments", []):
                for p in treatment.get("prescriptions", []):
                    values_per_key["med"].add(str(p.get("MedicatieStofnaam", "-")))
                    values_per_key["route"].add(str(p.get("ToedieningsRoute", "-")))
                    values_per_key["specialisme"].add(str(p.get("Specialisme", "-")))
                    values_per_key["groep"].add(str(p.get("Specialisme_groep", "-")))
                    values_per_key["setting"].add(str(p.get("community_or_hospital", "-")))
                    values_per_key["doel"].add(str(p.get("doel", "-")))
                    values_per_key["locatie"].add(str(p.get("locatie", "-")))
                    values_per_key["info"].add(str(p.get("aanvullende_informatie", "-")))
                for c in treatment.get("treatment_cultures", []):
                    values_per_key["materiaal"].add(str(c["materiaal_catCustom"]))
                    values_per_key["uitslag"].add(str(c["kweek_uitslagDef"]))
                    genus_val = c.get("microbe_genus")
                    values_per_key["genus"].add(str(genus_val) if genus_val else "NA")
                    values_per_key["microbe_cat"].add(str(c["microbe_catCustom"]))


    encoders = {}
    for key, values in tqdm(values_per_key.items(), desc="Fitting encoders"):
        le = LabelEncoder()
        le.fit(sorted(values))
        encoders[key] = le
    return encoders