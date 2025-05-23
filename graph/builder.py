from graph.features import normalize_date, safe_stats, encode
from typing import Dict, Tuple
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import pandas as pd 
from datetime import timedelta
import torch

def build_pyg_graph_with_all_features(treatment: dict, measurements: dict, label_encoders: Dict[str, LabelEncoder]) -> Tuple[Data, dict]:
    node_features = []
    edge_index = [[], []]
    edge_types = []
    node_types = []
    node_metadata = {}
    node_idx = {}

    node_type_to_idx = {"day": 0, "prescription": 1, "culture": 2, "measurement": 3}
    edge_type_to_idx = {"temporal": 0, "prescribed": 1, "culture": 2, "measurement": 3}


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
            encode(p["MedicatieStofnaam"], "med"),
            encode(p["ToedieningsRoute"], "route"),
            encode(p["Specialisme"], "specialisme"),
            encode(p["Specialisme_groep"], "groep"),
            encode(p["community_or_hospital"], "setting"),
            encode(p["doel"], "doel"),
            encode(p["locatie"], "locatie"),
            encode(p.get("aanvullende_informatie") or "-", "info")
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

    # === CULTURE nodes ===
    for i, c in enumerate(treatment.get("treatment_cultures", [])):
        node_id = f"culture_{i}"
        idx = len(node_features)
        node_idx[node_id] = idx
        genus_val = c.get("microbe_genus")
        feat = [
            encode(c["materiaal_catCustom"], "materiaal"),
            encode(c["kweek_uitslagDef"], "uitslag"),
            encode(str(genus_val) if genus_val else "NA", "genus"),
            encode(c["microbe_catCustom"], "microbe_cat")
        ]
        node_features.append(feat)
        node_types.append("culture")
        node_metadata[idx] = {"type": "culture", "afnamedatumtijd": c["afnamedatumtijd"]}

        c_day = normalize_date(c["afnamedatumtijd"])
        d_id = f"day_{c_day.date()}"
        if d_id in node_idx:
            edge_index[0].append(node_idx[d_id])
            edge_index[1].append(idx)
            edge_types.append("culture")

    # === MEASUREMENT nodes (optional: for detailed stats) ===
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
        node_metadata[idx] = {"type": "measurement", "date": str(d)}

        d_id = f"day_{d}"
        if d_id in node_idx:
            edge_index[0].append(node_idx[d_id])
            edge_index[1].append(idx)
            edge_types.append("measurement")

    # === Pad Features ===
    max_len = max(len(feat) for feat in node_features)
    for i, feat in enumerate(node_features):
        if len(feat) < max_len:
            node_features[i] = feat + [0.0] * (max_len - len(feat))

    # === Convert to PyG Data ===
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # y will be set later when creating the sample

    try:
        node_type_ids = torch.tensor([node_type_to_idx[nt] for nt in node_types], dtype=torch.long)
    except KeyError as e:
        print("[ERROR] Unknown node type:", e)
        print("Offending node_types list:", node_types)
        raise

    edge_type_ids = torch.tensor([edge_type_to_idx[et] for et in edge_types], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        node_type_ids=node_type_ids,
        edge_type_ids=edge_type_ids
    ), {
        "num_nodes": x.shape[0],
        "num_edges": edge_index.shape[0],
        "node_features": node_features,
        "node_types": node_types,
        "edge_types": edge_types,
        "node_metadata": node_metadata,
        "node_type_ids": node_type_ids.tolist(),
        "edge_type_ids": edge_type_ids.tolist()
    }