# pipeline/sample_generation.py

import torch
import pandas as pd
from tqdm import tqdm

from config import SAMPLE_CONFIG, PATHS
from data.io import load_json_subset, load_json_subset_only_iv
from data.labeling import label_iv_oral_switch_windows, earliest_switch_class
from data.preprocessing import cut_treatment_data_up_to_day_n, build_global_label_encoders
from graph.builder import build_pyg_graph_with_all_features
from config import MODEL_CONFIG

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
        return samples, df_meta, encoders

    print("Generating new samples…")
    data_loader = load_json_subset_only_iv if SAMPLE_CONFIG.only_iv else load_json_subset
    data = data_loader(PATHS.data_json, limit=SAMPLE_CONFIG.limit, balanced=SAMPLE_CONFIG.balanced)

    encoders = build_global_label_encoders(data)
    rows = []
    for pid, pat in data.items():
        for adm in pat["admissions"]:
            for tr in adm["treatments"]:
                for w in label_iv_oral_switch_windows(tr):
                    w.update({"pid": pid, "aid": adm["PatientContactId"], "tid": tr["treatment_id"]})
                    rows.append(w)

    samples, df_rows = [], []
    for r in tqdm(rows, desc="Creating samples"):
        cut = cut_treatment_data_up_to_day_n(data, r["pid"], r["aid"], r["tid"], r["day_n"])
        if cut is None:
            continue

        graph, _ = build_pyg_graph_with_all_features(
            cut["treatments"][0], cut.get("measurements", {}), encoders
        )

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
