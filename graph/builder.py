from graph.features import normalize_date, safe_stats, encode
from typing import Dict, Tuple
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import pandas as pd
from datetime import timedelta
import torch

def build_pyg_graph_with_all_features_1(treatment: dict, measurements: dict, label_encoders: Dict[str, LabelEncoder]) -> Tuple[Data, dict]:
    import torch
    from torch_geometric.data import Data
    from datetime import timedelta
    from graph.features import normalize_date, safe_stats, encode
    import pandas as pd

    node_features = []
    edge_index = [[], []]
    edge_types = []
    node_types = []
    node_metadata = {}
    node_idx = {}

    node_type_to_idx = {"day": 0, "prescription": 1, "culture": 2, "measurement": 3}
    edge_type_to_idx = {"temporal": 0, "prescribed": 1, "culture": 2, "measurement": 3}

    used_node_types = set()

    # Build vitals map for enrichment
    vitals_by_day = {}
    for key, entries in measurements.items():
        for e in entries:
            dt = normalize_date(e["MeetMoment"]).date()
            vitals_by_day.setdefault(dt, {}).setdefault(key, []).append(e)

    # Build a quick route map
    daily_routes = {}
    for p in treatment.get("prescriptions", []):
        s, e = pd.to_datetime(p["StartDatumTijd"]), pd.to_datetime(p["StopDatumTijd"])
        route = str(p.get("ToedieningsRoute", "")).lower()
        rtype = "iv" if "intraveneus" in route else "oraal"
        d = s.normalize()
        while d <= e.normalize():
            daily_routes.setdefault(d.date(), []).append(rtype)
            d += timedelta(days=1)

    # Culture map
    culture_days = {
        normalize_date(c["afnamedatumtijd"]).date(): True
        for c in treatment.get("treatment_cultures", [])
    }

    # === DAY nodes with enriched features ===
    start_date = normalize_date(treatment["treatment_start"])
    end_date = normalize_date(treatment["treatment_end"])
    days = pd.date_range(start=start_date, end=end_date)

    for i, day in enumerate(days):
        d = day.date()
        iv_count = daily_routes.get(d, []).count("iv")
        oral_count = daily_routes.get(d, []).count("oraal")

        vitals = vitals_by_day.get(d, {})
        hr = safe_stats([v.get("Hartfrequentie") for v in vitals.get("bpm", [])])
        o2 = safe_stats([v.get("o2Saturatie") for v in vitals.get("o2", [])])
        temp = safe_stats([v.get("Temperatuur") for v in vitals.get("temp", [])])
        crp = safe_stats([v.get("UitslagNumeriek") for v in vitals.get("crp", [])])

        features = [i, iv_count, oral_count] + crp + hr + o2 + temp + [int(d in culture_days)]

        node_id = f"day_{d}"
        idx = len(node_features)
        node_idx[node_id] = idx
        node_features.append(features)
        node_types.append("day")
        used_node_types.add("day")
        node_metadata[idx] = {"type": "day", "date": str(d)}

        if i > 0:
            prev_id = f"day_{days[i - 1].date()}"
            edge_index[0].append(node_idx[prev_id])
            edge_index[1].append(node_idx[node_id])
            edge_types.append("temporal")

    # === PRESCRIPTION nodes ===
    for p in treatment.get("prescriptions", []):
        node_id = f"presc_{p['OrderId']}"
        idx = len(node_features)
        node_idx[node_id] = idx
        feat = [
            (pd.to_datetime(p["StopDatumTijd"]) - pd.to_datetime(p["StartDatumTijd"])).total_seconds() / 3600.0,
            encode(label_encoders, p["MedicatieStofnaam"], "med"),
            encode(label_encoders, p["ToedieningsRoute"], "route"),
            encode(label_encoders, p["Specialisme"], "specialisme"),
            encode(label_encoders, p["Specialisme_groep"], "groep"),
            encode(label_encoders, p["community_or_hospital"], "setting"),
            encode(label_encoders, p["doel"], "doel"),
            encode(label_encoders, p["locatie"], "locatie"),
            encode(label_encoders, p.get("aanvullende_informatie") or "-", "info")
        ]
        node_features.append(feat)
        node_types.append("prescription")
        used_node_types.add("prescription")
        node_metadata[idx] = {"type": "prescription", "order_id": p["OrderId"]}

        p_start = normalize_date(p["StartDatumTijd"])
        p_stop = normalize_date(p["StopDatumTijd"])
        for d in pd.date_range(p_start, p_stop):
            d_id = f"day_{d.date()}"
            if d_id in node_idx:
                edge_index[0].append(node_idx[d_id])
                edge_index[1].append(idx)
                edge_types.append("prescribed")

    # === CULTURE nodes ===
    for i, c in enumerate(treatment.get("treatment_cultures", [])):
        node_id = f"culture_{i}"
        idx = len(node_features)
        node_idx[node_id] = idx
        genus_val = c.get("microbe_genus")
        feat = [
            encode(label_encoders, c["materiaal_catCustom"], "materiaal"),
            encode(label_encoders, c["kweek_uitslagDef"], "uitslag"),
            encode(label_encoders, str(genus_val) if genus_val else "NA", "genus"),
            encode(label_encoders, c["microbe_catCustom"], "microbe_cat")
        ]
        node_features.append(feat)
        node_types.append("culture")
        used_node_types.add("culture")
        node_metadata[idx] = {"type": "culture", "afnamedatumtijd": c["afnamedatumtijd"]}

        c_day = normalize_date(c["afnamedatumtijd"])
        d_id = f"day_{c_day.date()}"
        if d_id in node_idx:
            edge_index[0].append(node_idx[d_id])
            edge_index[1].append(idx)
            edge_types.append("culture")

    # === MEASUREMENT nodes (optional: for detailed stats) ===
    measurement_node_count = 0
    for day in days:
        d = day.date()
        if d not in vitals_by_day:
            continue
        vals = vitals_by_day[d]
        hr = safe_stats([v.get("Hartfrequentie") for v in vals.get("bpm", [])])
        o2 = safe_stats([v.get("o2Saturatie") for v in vals.get("o2", [])])
        temp = safe_stats([v.get("Temperatuur") for v in vals.get("temp", [])])
        crp = safe_stats([v.get("UitslagNumeriek") for v in vals.get("crp", [])])

        node_id = f"meas_{d}"
        idx = len(node_features)
        node_idx[node_id] = idx
        node_features.append(hr + o2 + temp + crp)
        node_types.append("measurement")
        used_node_types.add("measurement")
        node_metadata[idx] = {"type": "measurement", "date": str(d)}
        measurement_node_count += 1

        d_id = f"day_{d}"
        if d_id in node_idx:
            edge_index[0].append(node_idx[d_id])
            edge_index[1].append(idx)
            edge_types.append("measurement")

    # print(f"[DEBUG] Used node types: {sorted(used_node_types)}")
    # print(f"[DEBUG] Total nodes: {len(node_features)} | Measurement nodes: {measurement_node_count}")

    # === Pad Features ===
    max_len = max(len(feat) for feat in node_features)
    for i, feat in enumerate(node_features):
        if len(feat) < max_len:
            node_features[i] = feat + [0.0] * (max_len - len(feat))

    # === Convert to PyG Data ===
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    try:
        node_type_ids = torch.tensor([node_type_to_idx[nt] for nt in node_types], dtype=torch.long)
    except KeyError as e:
        print("[ERROR] Unknown node type:", e)
        print("Offending node_types list:", node_types)
        raise

    edge_type_ids = torch.tensor([edge_type_to_idx[et] for et in edge_types], dtype=torch.long)

    if torch.isnan(x).any():
        print("[FATAL] NaNs in node features")
    if torch.isinf(x).any():
        print("[FATAL] Infs in node features")
    if edge_index.max() >= x.shape[0] or edge_index.min() < 0:
        print("[FATAL] Invalid edge_index range")

    return Data(
        x=x,
        edge_index=edge_index,
        node_type_ids=node_type_ids,
        edge_type_ids=edge_type_ids
    ), {
        "num_nodes": x.shape[0],
        "num_edges": edge_index.shape[1],
        "node_features": node_features,
        "node_types": node_types,
        "edge_types": edge_types,
        "node_metadata": node_metadata,
        "node_type_ids": node_type_ids.tolist(),
        "edge_type_ids": edge_type_ids.tolist()
    }


def build_pyg_graph_with_all_features_2(treatment: dict, measurements: dict, label_encoders: Dict[str, LabelEncoder]) -> Tuple[Data, dict]:
    node_features = []
    edge_index = [[], []]
    edge_types = []
    node_types = []
    node_metadata = {}
    node_idx = {}

    node_type_to_idx = {"day": 0, "prescription": 1, "culture": 2, "measurement": 3, "treatment": 4}
    edge_type_to_idx = {"temporal": 0, "prescribed": 1, "culture": 2, "measurement": 3, "in_treatment": 4, "belongs_to": 5, "sampled_in": 6}

    ORAL_BIOAVAILABLE = {
        "AMOXICILLINE/CLAVULAANZUUR", "CIPROFLOXACINE", "CLINDAMYCINE", "COTRIMOXAZOL",
        "FLUCLOXACILLINE", "LEVOFLOXACINE", "LINEZOLIDE", "METRONIDAZOL", "MOXIFLOXACINE"
    }

    vitals_by_day = {}
    for key, entries in measurements.items():
        for e in entries:
            dt = normalize_date(e["MeetMoment"]).date()
            vitals_by_day.setdefault(dt, {}).setdefault(key, []).append(e)

    daily_routes = {}
    daily_meds = {}
    for p in treatment.get("prescriptions", []):
        s, e = pd.to_datetime(p["StartDatumTijd"]), pd.to_datetime(p["StopDatumTijd"])
        route = str(p.get("ToedieningsRoute", "")).lower()
        rtype = "iv" if "intraveneus" in route else "oraal"
        d = s.normalize()
        while d <= e.normalize():
            daily_routes.setdefault(d.date(), []).append(rtype)
            daily_meds.setdefault(d.date(), []).append(p["MedicatieStofnaam"])
            d += timedelta(days=1)

    treatment_node_id = f"treatment_{treatment['treatment_id']}"
    treat_idx = len(node_features)
    treatment_start = normalize_date(treatment["treatment_start"])
    treatment_days = pd.date_range(treatment_start, normalize_date(treatment["treatment_end"]))

    treat_feat = [
        encode(label_encoders, treatment.get("locatie", "-"), "locatie"),
        encode(label_encoders, treatment.get("doel", "-"), "doel"),
        len(treatment.get("prescriptions", []))
    ]
    node_features.append(treat_feat)
    node_types.append("treatment")
    node_idx[treatment_node_id] = treat_idx
    node_metadata[treat_idx] = {"type": "treatment", "treatment_id": treatment["treatment_id"]}

    culture_days = {normalize_date(c["afnamedatumtijd"]).date(): True for c in treatment.get("treatment_cultures", [])}

    temp_history = []
    crp_history = []
    fever_threshold = 38.3

    for i, day in enumerate(treatment_days):
        d = day.date()
        vitals = vitals_by_day.get(d, {})
        hr = safe_stats([v.get("Hartfrequentie") for v in vitals.get("bpm", [])])
        o2 = safe_stats([v.get("o2Saturatie") for v in vitals.get("o2", [])])
        temp = safe_stats([v.get("Temperatuur") for v in vitals.get("temp", [])])
        crp = safe_stats([v.get("UitslagNumeriek") for v in vitals.get("crp", [])])

        clean_temp_vals = [v.get("Temperatuur") for v in vitals.get("temp", []) if v.get("Temperatuur") is not None]
        clean_crp_vals = [v.get("UitslagNumeriek") for v in vitals.get("crp", []) if v.get("UitslagNumeriek") is not None]

        max_temp = max(clean_temp_vals) if clean_temp_vals else 0
        mean_crp = sum(clean_crp_vals) / len(clean_crp_vals) if clean_crp_vals else 0

        delta_temp = (max_temp - temp_history[-1]) if len(temp_history) else 0
        delta_crp = (mean_crp - crp_history[-1]) if len(crp_history) else 0
        fever_free_days = sum(1 for t in reversed(temp_history) if t < fever_threshold)

        temp_history.append(max_temp)
        crp_history.append(mean_crp)

        iv_count = daily_routes.get(d, []).count("iv")
        oral_count = daily_routes.get(d, []).count("oraal")
        distinct_abx = len(set(daily_meds.get(d, [])))

        features = [
            i,  # day index
            iv_count,
            oral_count,
            delta_crp,
            delta_temp,
            fever_free_days,
            int(d in culture_days),
            distinct_abx,
            d.weekday(),  # day of week
        ] + crp + hr + o2 + temp

        node_id = f"day_{d}"
        idx = len(node_features)
        node_idx[node_id] = idx
        node_features.append(features)
        node_types.append("day")
        node_metadata[idx] = {"type": "day", "date": str(d)}

        if i > 0:
            prev_id = f"day_{treatment_days[i - 1].date()}"
            edge_index[0].append(node_idx[prev_id])
            edge_index[1].append(idx)
            edge_types.append("temporal")

        edge_index[0].append(idx)
        edge_index[1].append(treat_idx)
        edge_types.append("in_treatment")

    for p in treatment.get("prescriptions", []):
        node_id = f"presc_{p['OrderId']}"
        idx = len(node_features)
        node_idx[node_id] = idx

        duration_hr = (pd.to_datetime(p["StopDatumTijd"]) - pd.to_datetime(p["StartDatumTijd"])).total_seconds() / 3600.0
        med = p["MedicatieStofnaam"]
        route = p["ToedieningsRoute"]
        is_oral_bioavailable = int(med in ORAL_BIOAVAILABLE and route.lower().startswith("oraal"))

        feat = [
            duration_hr,
            encode(label_encoders, med, "med"),
            encode(label_encoders, route, "route"),
            encode(label_encoders, p["Specialisme"], "specialisme"),
            encode(label_encoders, p["Specialisme_groep"], "groep"),
            encode(label_encoders, p["community_or_hospital"], "setting"),
            encode(label_encoders, p["doel"], "doel"),
            encode(label_encoders, p["locatie"], "locatie"),
            encode(label_encoders, p.get("aanvullende_informatie") or "-", "info"),
            is_oral_bioavailable
        ]

        node_features.append(feat)
        node_types.append("prescription")
        node_metadata[idx] = {"type": "prescription", "order_id": p["OrderId"]}

        p_start = normalize_date(p["StartDatumTijd"])
        p_stop = normalize_date(p["StopDatumTijd"])
        for d in pd.date_range(p_start, p_stop):
            d_id = f"day_{d.date()}"
            if d_id in node_idx:
                edge_index[0].append(node_idx[d_id])
                edge_index[1].append(idx)
                edge_types.append("prescribed")
        edge_index[0].append(idx)
        edge_index[1].append(treat_idx)
        edge_types.append("belongs_to")

    for i, c in enumerate(treatment.get("treatment_cultures", [])):
        node_id = f"culture_{i}"
        idx = len(node_features)
        node_idx[node_id] = idx
        genus_val = c.get("microbe_genus")

        culture_time = normalize_date(c["afnamedatumtijd"])
        delta_days = (culture_time - treatment_start).days
        result = c["kweek_uitslagDef"]
        is_positive_recent = int(result == "pos" and delta_days >= 0 and delta_days <= 3)

        feat = [
            encode(label_encoders, c["materiaal_catCustom"], "materiaal"),
            encode(label_encoders, result, "uitslag"),
            encode(label_encoders, str(genus_val) if genus_val else "NA", "genus"),
            encode(label_encoders, c["microbe_catCustom"], "microbe_cat"),
            delta_days,
            is_positive_recent
        ]
        node_features.append(feat)
        node_types.append("culture")
        node_metadata[idx] = {"type": "culture", "afnamedatumtijd": c["afnamedatumtijd"]}

        d_id = f"day_{culture_time.date()}"
        if d_id in node_idx:
            edge_index[0].append(node_idx[d_id])
            edge_index[1].append(idx)
            edge_types.append("culture")

        edge_index[0].append(idx)
        edge_index[1].append(treat_idx)
        edge_types.append("sampled_in")

    for day in treatment_days:
        d = day.date()
        if d not in vitals_by_day:
            continue
        vals = vitals_by_day[d]
        hr = safe_stats([v.get("Hartfrequentie") for v in vals.get("bpm", [])])
        o2 = safe_stats([v.get("o2Saturatie") for v in vals.get("o2", [])])
        temp = safe_stats([v.get("Temperatuur") for v in vals.get("temp", [])])
        crp = safe_stats([v.get("UitslagNumeriek") for v in vals.get("crp", [])])

        node_id = f"meas_{d}"
        idx = len(node_features)
        node_idx[node_id] = idx
        node_features.append(hr + o2 + temp + crp)
        node_types.append("measurement")
        node_metadata[idx] = {"type": "measurement", "date": str(d)}

        d_id = f"day_{d}"
        if d_id in node_idx:
            edge_index[0].append(node_idx[d_id])
            edge_index[1].append(idx)
            edge_types.append("measurement")

    max_len = max(len(feat) for feat in node_features)
    for i, feat in enumerate(node_features):
        if len(feat) < max_len:
            node_features[i] = feat + [0.0] * (max_len - len(feat))

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_type_ids = torch.tensor([node_type_to_idx[nt] for nt in node_types], dtype=torch.long)
    edge_type_ids = torch.tensor([edge_type_to_idx[et] for et in edge_types], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        node_type_ids=node_type_ids,
        edge_type_ids=edge_type_ids
    ), {
        "num_nodes": x.shape[0],
        "num_edges": edge_index.shape[1],
        "node_features": node_features,
        "node_types": node_types,
        "edge_types": edge_types,
        "node_metadata": node_metadata,
        "node_type_ids": node_type_ids.tolist(),
        "edge_type_ids": edge_type_ids.tolist()
    }

# ---------------------------------------------------------------------
# Tunable domain constants – edit here if you want to refine criteria
# ---------------------------------------------------------------------
FEVER_THRESHOLD = 38.3                                  # °C
GOOD_BIOAVAIL_MEDS = {
    "AMOXICILLINE/CLAVULAANZUUR", "CIPROFLOXACINE",
    "CLINDAMYCINE", "COTRIMOXAZOL", "FLUCLOXACILLINE",
    "LINEZOLID", "METRONIDAZOL"
}
# crude mapping from infection-site strings (lower-case Dutch) to
# culture materiaal categories; expand if you have more labels
SITE_TO_MATERIAL = {
    "urine": "urine",
    "urineweginfectie": "urine",
    "gynaecologisch": "gynecologisch",
    "gynaecologische": "gynecologisch",
    "intra-abdominale": "bloed",
    "sepsis": "bloed"
}
# ---------------------------------------------------------------------

def build_pyg_graph_with_all_features_3(
    treatment: dict,
    measurements: dict,
    label_encoders: Dict[str, LabelEncoder]
) -> Tuple[Data, dict]:
    """
    Build a PyG graph for a single treatment episode *up to and including
    the final day present in `treatment["treatment_end"]`*.

    The function adds:
      • richer DAY-node temporal features (deltas, fever-free streak, etc.)
      • a TREATMENT node with “so-far” aggregates (no future leakage)
      • prescription-level eligibility / bio-availability flags
      • culture recency & site-matching features
      • new edge types linking everything together

    Only information that would be known **up to the graph’s last day** is
    included.  If you want a day-*n* snapshot, slice the treatment JSON
    before calling this function (i.e. cap `treatment["treatment_end"]`
    and remove future prescriptions/cultures).
    """
    # ------------------------------------------------------------------
    # initialise holders
    # ------------------------------------------------------------------
    node_features, node_types, node_metadata = [], [], {}
    edge_index, edge_types = [[], []], []
    node_idx = {}

    node_type_to_idx = {
        "day": 0, "prescription": 1, "culture": 2,
        "measurement": 3, "treatment": 4
    }
    edge_type_to_idx = {
        "temporal": 0, "prescribed": 1, "culture": 2,
        "measurement": 3, "in_treatment": 4,
        "belongs_to": 5, "sampled_in": 6
    }

    # ------------------------------------------------------------------
    # build helper maps
    # ------------------------------------------------------------------
    # 1️⃣ vitals per calendar-day ------------------------------------------------
    vitals_by_day = {}
    for key, entries in measurements.items():
        for e in entries:
            dt = normalize_date(e["MeetMoment"]).date()
            vitals_by_day.setdefault(dt, {}).setdefault(key, []).append(e)

    # 2️⃣ IV / ORAL exposure per day -------------------------------------------
    daily_routes = {}
    for p in treatment.get("prescriptions", []):
        s, e = pd.to_datetime(p["StartDatumTijd"]), pd.to_datetime(p["StopDatumTijd"])
        route = str(p.get("ToedieningsRoute", "")).lower()
        rtype = "iv" if "intraveneus" in route else "oraal"
        d = s.normalize()
        while d <= e.normalize():
            daily_routes.setdefault(d.date(), []).append(rtype)
            d += timedelta(days=1)

    # 3️⃣ culture collection days ----------------------------------------------
    culture_days = {
        normalize_date(c["afnamedatumtijd"]).date(): True
        for c in treatment.get("treatment_cultures", [])
    }

    # ------------------------------------------------------------------
    # DAY nodes – temporal stats & trends
    # ------------------------------------------------------------------
    start_date = normalize_date(treatment["treatment_start"])
    end_date   = normalize_date(treatment["treatment_end"])
    days = pd.date_range(start=start_date, end=end_date)

    temp_history_max, crp_history_mean, o2_history_mean = [], [], []

    for i, day in enumerate(days):
        d = day.date()

        # counts of IV / ORAL prescriptions active today
        iv_count   = daily_routes.get(d, []).count("iv")
        oral_count = daily_routes.get(d, []).count("oraal")

        # vitals today
        vitals  = vitals_by_day.get(d, {})
        hr_vals = [v.get("Hartfrequentie")   for v in vitals.get("bpm",  [])]
        o2_vals = [v.get("o2Saturatie")      for v in vitals.get("o2",   [])]
        t_vals  = [v.get("Temperatuur")      for v in vitals.get("temp", [])]
        crp_vals= [v.get("UitslagNumeriek")  for v in vitals.get("crp",  [])]

        hr  = safe_stats(hr_vals)
        o2  = safe_stats(o2_vals)
        tmp = safe_stats(t_vals)
        crp = safe_stats(crp_vals)

        # ---------- temporal deltas ----------
        clean_t_vals  = [v for v in t_vals if v is not None]
        clean_crp_vals = [v for v in crp_vals if v is not None]
        clean_o2_vals = [v for v in o2_vals if v is not None]

        max_temp_today = max(clean_t_vals) if clean_t_vals else 0.0
        mean_crp_today = sum(clean_crp_vals) / len(clean_crp_vals) if clean_crp_vals else 0.0
        mean_o2_today  = sum(clean_o2_vals) / len(clean_o2_vals)   if clean_o2_vals else 0.0


        delta_temp = (max_temp_today - temp_history_max[-1]) if temp_history_max else 0.0
        delta_crp  = (mean_crp_today - crp_history_mean[-1]) if crp_history_mean else 0.0
        delta_o2   = (mean_o2_today  - o2_history_mean[-1])  if o2_history_mean  else 0.0

        temp_history_max.append(max_temp_today)
        crp_history_mean.append(mean_crp_today)
        o2_history_mean.append(mean_o2_today)

        # fever-free streak
        fever_free_days = 0
        for t in reversed(temp_history_max[:-1]):          # exclude today
            if t < FEVER_THRESHOLD:
                fever_free_days += 1
            else:
                break

        # days since high CRP (>100).  Use ∞ (=-1) if never occurred.
        days_since_high_crp = -1
        for idx_back, val in enumerate(reversed(crp_history_mean[:-1])):
            if val > 100:
                days_since_high_crp = idx_back
                break

        # active antibiotics
        active_abx = set()
        for p in treatment.get("prescriptions", []):
            s = pd.to_datetime(p["StartDatumTijd"]).normalize().date()
            e = pd.to_datetime(p["StopDatumTijd"]).normalize().date()
            if s <= d <= e:
                active_abx.add(p["MedicatieStofnaam"])
        active_abx_count = len(active_abx)

        # assemble feature vector for the DAY node ------------------------------
        features = [
            i,                                  # 0: absolute day index
            iv_count, oral_count,               # 1-2: route counts
            delta_temp, delta_crp, delta_o2,    # 3-5: trends
            fever_free_days,                    # 6
            days_since_high_crp,                # 7
            active_abx_count,                   # 8
            d.weekday(),                        # 9: week-day
            int(d in culture_days)              # 10: culture collected?
        ] + crp + hr + o2 + tmp                # add raw stats

        node_id = f"day_{d}"
        idx = len(node_features)
        node_idx[node_id] = idx
        node_features.append(features)
        node_types.append("day")
        node_metadata[idx] = {"type": "day", "date": str(d)}

        # temporal edge to previous day
        if i > 0:
            prev_id = f"day_{days[i - 1].date()}"
            edge_index[0].append(node_idx[prev_id])
            edge_index[1].append(idx)
            edge_types.append("temporal")

    # ------------------------------------------------------------------
    # TREATMENT node  (aggregated “so-far” info)
    # ------------------------------------------------------------------
    graph_today = days[-1].date()      # last day included in this graph

    treatment_id = treatment["treatment_id"]
    t_node_id = f"treat_{treatment_id}"
    t_idx = len(node_features)

    # counts up to graph_today
    n_days_so_far   = len(days)
    n_iv_days       = sum(daily_routes.get(d.date(), []).count("iv")   > 0 for d in days)
    n_oral_days     = sum(daily_routes.get(d.date(), []).count("oraal")> 0 for d in days)

    # distinct antibiotics & last route
    abx_routes_so_far, last_route = set(), 0
    for p in treatment.get("prescriptions", []):
        s = pd.to_datetime(p["StartDatumTijd"]).normalize().date()
        if s <= graph_today:
            abx_routes_so_far.add(p["MedicatieStofnaam"])
            if pd.to_datetime(p["StartDatumTijd"]).normalize().date() <= graph_today <= pd.to_datetime(p["StopDatumTijd"]).normalize().date():
                last_route = 1 if "oraal" in p["ToedieningsRoute"].lower() else 0

    t_features = [
        n_days_so_far, n_iv_days, n_oral_days,
        len(abx_routes_so_far),                # distinct antibiotics
        last_route                             # 0=IV, 1=oral
    ]

    node_idx[t_node_id] = t_idx
    node_features.append(t_features)
    node_types.append("treatment")
    node_metadata[t_idx] = {"type": "treatment", "treatment_id": treatment_id}

    # connect DAY → TREATMENT (“in_treatment”)
    for d in days:
        edge_index[0].append(node_idx[f"day_{d.date()}"])
        edge_index[1].append(t_idx)
        edge_types.append("in_treatment")

    # ------------------------------------------------------------------
    # PRESCRIPTION nodes
    # ------------------------------------------------------------------
    for p in treatment.get("prescriptions", []):
        p_id  = p['OrderId']
        node_id = f"presc_{p_id}"
        idx     = len(node_features)
        node_idx[node_id] = idx

        p_start = pd.to_datetime(p["StartDatumTijd"])
        p_stop  = pd.to_datetime(p["StopDatumTijd"])

        # active on graph_today?
        active_today = int(p_start.normalize().date() <= graph_today <= p_stop.normalize().date())

        # time since last dose (days; 0 if still active)
        last_dose_delta = (graph_today - p_stop.normalize().date()).days
        last_dose_delta = max(0, last_dose_delta)

        # current duration so far (days)
        current_dur = (min(graph_today, p_stop.normalize().date()) - p_start.normalize().date()).days + 1
        current_dur = max(0, current_dur)

        oral_good_bio = int(
            "oraal" in p["ToedieningsRoute"].lower() and
            p["MedicatieStofnaam"] in GOOD_BIOAVAIL_MEDS
        )

        feat = [
            current_dur,                        # 0: exposure so far (days)
            encode(label_encoders, p["MedicatieStofnaam"], "med"),
            encode(label_encoders, p["ToedieningsRoute"], "route"),
            encode(label_encoders, p["Specialisme"], "specialisme"),
            encode(label_encoders, p["Specialisme_groep"], "groep"),
            encode(label_encoders, p["community_or_hospital"], "setting"),
            encode(label_encoders, p["doel"], "doel"),
            encode(label_encoders, p["locatie"], "locatie"),
            encode(label_encoders, p.get("aanvullende_informatie") or "-", "info"),
            active_today,                       # 9
            oral_good_bio,                      # 10
            last_dose_delta                     # 11
        ]
        node_features.append(feat)
        node_types.append("prescription")
        node_metadata[idx] = {"type": "prescription", "order_id": p_id}

        # connect DAY → PRESCRIPTION (“prescribed”)
        for d in pd.date_range(p_start, p_stop):
            d_id = f"day_{d.date()}"
            if d_id in node_idx:
                edge_index[0].append(node_idx[d_id])
                edge_index[1].append(idx)
                edge_types.append("prescribed")

        # connect PRESCRIPTION → TREATMENT
        edge_index[0].append(idx)
        edge_index[1].append(t_idx)
        edge_types.append("belongs_to")

    # ------------------------------------------------------------------
    # CULTURE nodes
    # ------------------------------------------------------------------
    for i, c in enumerate(treatment.get("treatment_cultures", [])):
        node_id = f"culture_{i}"
        idx     = len(node_features)
        node_idx[node_id] = idx

        genus_val  = c.get("microbe_genus")
        mat_cat    = c["materiaal_catCustom"]

        # recency relative to graph_today
        c_day = normalize_date(c["afnamedatumtijd"]).date()
        days_since = (graph_today - c_day).days
        days_since = max(days_since, 0)

        # does culture site match infection site?
        loc_low = (treatment.get("locatie") or "").lower()
        match_mat = 0
        for k, v in SITE_TO_MATERIAL.items():
            if k in loc_low and v.lower() in mat_cat.lower():
                match_mat = 1
                break

        feat = [
            encode(label_encoders, mat_cat, "materiaal"),
            encode(label_encoders, c["kweek_uitslagDef"], "uitslag"),
            encode(label_encoders, str(genus_val) if genus_val else "NA", "genus"),
            encode(label_encoders, c["microbe_catCustom"], "microbe_cat"),
            days_since,                         # 4
            match_mat,                          # 5
            int(c["kweek_uitslagDef"] == "pos" and days_since <= 3)  # 6: recent positive?
        ]
        node_features.append(feat)
        node_types.append("culture")
        node_metadata[idx] = {"type": "culture", "afnamedatumtijd": c["afnamedatumtijd"]}

        # DAY → CULTURE
        d_id = f"day_{c_day}"
        if d_id in node_idx:
            edge_index[0].append(node_idx[d_id])
            edge_index[1].append(idx)
            edge_types.append("culture")

        # CULTURE → TREATMENT
        edge_index[0].append(idx)
        edge_index[1].append(t_idx)
        edge_types.append("sampled_in")

    # ------------------------------------------------------------------
    # MEASUREMENT nodes  (unchanged granularity, richer stats already on DAY)
    # ------------------------------------------------------------------
    for day in days:
        d = day.date()
        if d not in vitals_by_day:
            continue
        vals = vitals_by_day[d]
        hr   = safe_stats([v.get("Hartfrequentie")   for v in vals.get("bpm",  [])])
        o2   = safe_stats([v.get("o2Saturatie")      for v in vals.get("o2",   [])])
        t    = safe_stats([v.get("Temperatuur")      for v in vals.get("temp", [])])
        crp  = safe_stats([v.get("UitslagNumeriek")  for v in vals.get("crp",  [])])

        node_id = f"meas_{d}"
        idx     = len(node_features)
        node_idx[node_id] = idx
        node_features.append(hr + o2 + t + crp)
        node_types.append("measurement")
        node_metadata[idx] = {"type": "measurement", "date": str(d)}

        edge_index[0].append(node_idx[f"day_{d}"])
        edge_index[1].append(idx)
        edge_types.append("measurement")

    # ------------------------------------------------------------------
    # pad feature vectors so every node has equal length
    # ------------------------------------------------------------------
    max_len = max(len(f) for f in node_features)
    for i, f in enumerate(node_features):
        pad = max_len - len(f)
        if pad:
            node_features[i] = f + [0.0] * pad

    # ------------------------------------------------------------------
    # convert to tensors & return
    # ------------------------------------------------------------------
    x           = torch.tensor(node_features, dtype=torch.float)
    edge_index  = torch.tensor(edge_index,  dtype=torch.long)
    node_type_t = torch.tensor([node_type_to_idx[t] for t in node_types], dtype=torch.long)
    edge_type_t = torch.tensor([edge_type_to_idx[e] for e in edge_types], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        node_type_ids=node_type_t,
        edge_type_ids=edge_type_t
    )

    debug_info = {
        "num_nodes": x.shape[0],
        "num_edges": edge_index.shape[1],
        "node_features": node_features,
        "node_types": node_types,
        "edge_types": edge_types,
        "node_metadata": node_metadata,
        "node_type_ids": node_type_t.tolist(),
        "edge_type_ids": edge_type_t.tolist()
    }
    return data, debug_info