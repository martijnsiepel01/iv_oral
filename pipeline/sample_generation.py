# pipeline/sample_generation.py

import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path as path

from config import SAMPLE_CONFIG, PATHS
from data.io import load_json_subset, load_json_subset_only_iv
from data.labeling import label_iv_oral_switch_windows, earliest_switch_class
from data.preprocessing import cut_treatment_data_up_to_day_n, build_global_label_encoders
from graph.builder import build_pyg_graph_with_all_features_1, build_pyg_graph_with_all_features_2, build_pyg_graph_with_all_features_3
from config import MODEL_CONFIG

# def load_or_generate_samples():
#     # Force loading the specific samples and meta files
#     SAMPLE_CONFIG.samples_path = path(r"C:\Users\Martijn\OneDrive\PhD\iv_oral_v0.1\iv_oral_gnn\samples\old\samples_iv_oral_only_iv_5000.pt")    
#     SAMPLE_CONFIG.meta_path = path(r"C:\Users\Martijn\OneDrive\PhD\iv_oral_v0.1\iv_oral_gnn\samples\old\samples_iv_oral_only_iv_5000.parquet")

#     if SAMPLE_CONFIG.samples_path.exists() and SAMPLE_CONFIG.meta_path.exists():
#         print(f"Loading samples from {SAMPLE_CONFIG.samples_path}")
#         print(f"Loading metadata from {SAMPLE_CONFIG.meta_path}")
#         samples = torch.load(SAMPLE_CONFIG.samples_path, weights_only=False)
#         df_meta = pd.read_parquet(SAMPLE_CONFIG.meta_path)

#         # Relabel for binary classification if needed
#         if MODEL_CONFIG["binary"]:
#             print("Converting cached labels to binary...")
#             new_samples, new_df_rows = [], []
#             for (graph, _), row in zip(samples, df_meta.to_dict(orient="records")):
#                 binary_label = 1 if row["switch_day+1"] else 0
#                 new_samples.append((graph, torch.tensor(binary_label, dtype=torch.long)))

#                 row["no_switch_day+1"] = int(binary_label == 0)
#                 row["switch_day+1"] = int(binary_label == 1)
#                 row.pop("switch_day+2", None)
#                 row.pop("switch_day+3", None)
#                 new_df_rows.append(row)

#             samples = new_samples
#             df_meta = pd.DataFrame(new_df_rows)

#         # Load full data only to rebuild encoders
#         data_full = load_json_subset(PATHS.data_json, limit=None, balanced=False)
#         encoders = build_global_label_encoders(data_full)
#         print({key: len(enc.classes_) for key, enc in encoders.items()})
#         return samples, df_meta, encoders
#     else:
#         print("Sample or meta file not found, generating new samples...")
#         # You can uncomment and use your sample generation code here
#         raise FileNotFoundError(f"Sample or meta file not found at {SAMPLE_CONFIG.samples_path} or {SAMPLE_CONFIG.meta_path}")


def load_or_generate_samples():
    if SAMPLE_CONFIG.samples_path.exists() and SAMPLE_CONFIG.meta_path.exists():
        print(f"Loading samples from {SAMPLE_CONFIG.samples_path}")
        print(f"Loading metadata from {SAMPLE_CONFIG.meta_path}")
        samples = torch.load(SAMPLE_CONFIG.samples_path, weights_only=False)
        df_meta = pd.read_parquet(SAMPLE_CONFIG.meta_path)

        # Relabel for binary classification if needed
        if MODEL_CONFIG["binary"]:
            print("Converting cached labels to binary...")
            new_samples, new_df_rows = [], []
            for (graph, label), row in zip(samples, df_meta.to_dict(orient="records")):
                binary_label = 1 if row["switch_day+1"] else 0
                new_samples.append((graph, torch.tensor(binary_label, dtype=torch.long)))

                row["no_switch_day+1"] = int(binary_label == 0)
                row["switch_day+1"] = int(binary_label == 1)
                row.pop("switch_day+2", None)
                row.pop("switch_day+3", None)
                new_df_rows.append(row)

            samples = new_samples
            df_meta = pd.DataFrame(new_df_rows)

        # Load full data only to rebuild encoders
        data_full = load_json_subset(PATHS.data_json, limit=None, balanced=False)
        encoders = build_global_label_encoders(data_full)
        print({key: len(enc.classes_) for key, enc in encoders.items()})
        if SAMPLE_CONFIG.downsample:
            print("Applying downsampling to cached data")
            df_meta = df_meta.copy()
            df_meta["key"] = list(zip(df_meta["pid"], df_meta["aid"], df_meta["tid"]))
            downsampled_idxs = []

            for key, group in df_meta.groupby("key", sort=False):
                group = group.sort_values("day_n")
                no_switch_mask = group["switch_day+1"] == 0
                switch_mask = group["switch_day+1"] == 1

                no_switch_indices = group[no_switch_mask].index.tolist()
                keep_n = int(len(no_switch_indices) * 0.75)
                kept_no_switch = no_switch_indices[keep_n:]
                kept_switch = group[switch_mask].index.tolist()

                downsampled_idxs.extend(kept_no_switch + kept_switch)

            downsampled_idxs = sorted(downsampled_idxs)
            df_meta = df_meta.loc[downsampled_idxs].reset_index(drop=True)
            samples = [samples[i] for i in downsampled_idxs]
            print(f"→ Cached rows after downsampling: {len(samples)}")
        return samples, df_meta, encoders

    print("Generating new samples…")
    data_loader = load_json_subset_only_iv if SAMPLE_CONFIG.only_iv else load_json_subset
    data = data_loader(PATHS.data_json, limit=SAMPLE_CONFIG.limit, balanced=SAMPLE_CONFIG.balanced)

    data_full = load_json_subset(PATHS.data_json, limit=None, balanced=False)
    encoders = build_global_label_encoders(data_full)
    print({key: len(enc.classes_) for key, enc in encoders.items()})
    rows = []
    for pid, pat in data.items():
        for adm in pat["admissions"]:
            for tr in adm["treatments"]:
                for w in label_iv_oral_switch_windows(tr):
                    w.update({"pid": pid, "aid": adm["PatientContactId"], "tid": tr["treatment_id"]})
                    rows.append(w)

    if SAMPLE_CONFIG.downsample:
        print("Applying downsampling: removing first 50% of no-switch rows per treatment")
        grouped = {}
        for row in rows:
            key = (row["pid"], row["aid"], row["tid"])
            grouped.setdefault(key, []).append(row)

        filtered_rows = []
        for group_rows in grouped.values():
            # Sort by day_n to ensure chronological order
            group_rows.sort(key=lambda r: r["day_n"])
            non_switch_rows = [r for r in group_rows if not any(r[f"switch_on_day_n+{h}"] for h in range(1, 4))]
            switch_rows = [r for r in group_rows if any(r[f"switch_on_day_n+{h}"] for h in range(1, 4))]

            # Downsample: keep only last 25% of non-switch rows
            keep_n = int(len(non_switch_rows) * 0.75)
            filtered_rows.extend(non_switch_rows[keep_n:] + switch_rows)

        rows = filtered_rows
        print(f"→ Remaining rows after downsampling: {len(rows)}")

    samples, df_rows = [], []
    for r in tqdm(rows, desc="Creating samples"):
        cut = cut_treatment_data_up_to_day_n(data, r["pid"], r["aid"], r["tid"], r["day_n"])
        # meas = cut.get("measurements", {})
        # n_meas = sum(len(v) for v in meas.values())
        # if n_meas == 0:
        #     print(f"[NO MEAS] {r['pid']} | {r['aid']} | {r['day_n']} → 0 measurements")
        # expected_keys = ['bpm', 'o2', 'temp', 'crp']
        # missing = [k for k in expected_keys if not meas.get(k)]
        # if missing:
        #     print(f"[MISSING MEAS TYPES] {r['pid']} → {missing}")

        if cut is None:
            continue
        
        if SAMPLE_CONFIG.graph_type == 1:
            graph, _ = build_pyg_graph_with_all_features_1(
                cut["treatments"][0], cut.get("measurements", {}), encoders
            )
        elif SAMPLE_CONFIG.graph_type == 2:
            graph, _ = build_pyg_graph_with_all_features_2(
                cut["treatments"][0], cut.get("measurements", {}), encoders
            )
        elif SAMPLE_CONFIG.graph_type == 3:
            graph, _ = build_pyg_graph_with_all_features_3(
                cut["treatments"][0], cut.get("measurements", {}), encoders
            ) 
        else:
            raise ValueError("Invalid value for graph_construction_type")
            

        label_int = earliest_switch_class(r, MODEL_CONFIG["binary"])
        samples.append((graph, torch.tensor(label_int, dtype=torch.long)))

        row = {
            "pid": r["pid"], "aid": r["aid"], "tid": r["tid"], "day_n": r["day_n"],
            "no_switch_day+1": int(label_int == 0),
            "switch_day+1":    int(label_int == 1),
            "json_cut": cut,
        }

        if not MODEL_CONFIG["binary"]:
            row["switch_day+2"] = int(label_int == 2)
            row["switch_day+3"] = int(label_int == 3)

        df_rows.append(row)

    max_feat_len = max(g.x.shape[1] for g, _ in samples)
    for i, (graph, lbl) in enumerate(samples):
        if graph.x.shape[1] < max_feat_len:
            pad = torch.zeros(graph.x.shape[0], max_feat_len - graph.x.shape[1])
            graph.x = torch.cat([graph.x, pad], dim=1)
            samples[i] = (graph, lbl)

    df_meta = pd.DataFrame(df_rows)
    df_meta.to_parquet(SAMPLE_CONFIG.meta_path)
    torch.save(samples, SAMPLE_CONFIG.samples_path)

    print(f"Saved samples to {SAMPLE_CONFIG.samples_path}")
    print(f"Saved metadata to {SAMPLE_CONFIG.meta_path}")
    return samples, df_meta, encoders
