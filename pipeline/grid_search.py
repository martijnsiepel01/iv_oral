# pipeline/grid_search.py

import json
import csv
import copy
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from cli import parse_args, override_config_from_args
from pipeline.run_training import run

from pipeline.sample_generation import load_or_generate_samples
from config import MODEL_CONFIG, SAMPLE_CONFIG, PATHS

def load_param_grid(path: str):
    with open(path, "r") as f:
        grid_dict = json.load(f)
    return list(ParameterGrid(grid_dict))


# def grid_search(args):
#     if not args.param_grid:
#         raise ValueError("You must provide --param_grid <path_to_json>")

#     search_space = load_param_grid(args.param_grid)
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     results_csv = PATHS.models / f"grid_search_{timestamp}.csv"

#     print(f"\nLoaded grid with {len(search_space)} configurations.")
#     print(f"Saving results to: {results_csv}\n")

#     best_score = -1
#     best_cfg = None
#     first_run = True
#     metric_keys = []

#     for i, cfg in enumerate(search_space):
#         print(f"\n=== Config {i+1}/{len(search_space)} ===")
#         for k, v in cfg.items():
#             print(f"{k:15s}: {v}")
#         print("Training...")

#         args_copy = copy.deepcopy(args)
#         for k, v in cfg.items():
#             setattr(args_copy, k, v)

#         override_config_from_args(args_copy)
#         SAMPLE_CONFIG.num_epochs = args.search_epochs

#         metrics = run()

#         if first_run:
#             # Build the metric column headers once
#             base_keys = ["avg_loss", "f1_macro", "f1_micro", "auroc"]
#             per_class_keys = []
#             for i in range(len(metrics["f1_per_class"])):
#                 per_class_keys.extend([
#                     f"f1_class_{i}",
#                     f"precision_class_{i}",
#                     f"recall_class_{i}"
#                 ])
#             metric_keys = base_keys + per_class_keys
#             csv_keys = list(cfg.keys()) + metric_keys

#             with open(results_csv, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(csv_keys)

#             first_run = False

#         # Flatten metrics into a row
#         metric_row = [
#             metrics["avg_loss"],
#             metrics["f1_macro"],
#             metrics["f1_micro"],
#             metrics["auroc"]
#         ] + list(metrics["f1_per_class"]) + list(metrics["precision_per_class"]) + list(metrics["recall_per_class"])

#         with open(results_csv, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([cfg[k] for k in cfg] + metric_row)

#         f1 = metrics["f1_macro"]
#         print(f"→ F1_macro: {f1:.4f}")

#         if f1 > best_score:
#             best_score = f1
#             best_cfg = cfg

#     # Summary
#     print("\n==== Best configuration ====")
#     for k, v in best_cfg.items():
#         print(f"{k:15s}: {v}")
#     print(f"{'F1-macro':15s}: {best_score:.4f}")
#     print(f"\nAll results saved to: {results_csv}")

def line_search(args):
    if not args.param_grid:
        raise ValueError("You must provide --param_grid <path_to_json>")

    with open(args.param_grid, "r") as f:
        grid_dict = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_csv = PATHS.models / f"line_search_{timestamp}.csv"

    print(f"\nRunning line search on {len(grid_dict)} parameters.")
    print(f"Saving results to: {results_csv}\n")

    # Ensure default values are in config before loading data
    MODEL_CONFIG["binary"] = args.binary
    SAMPLE_CONFIG.limit = args.limit
    SAMPLE_CONFIG.balanced = not args.no_balance
    SAMPLE_CONFIG.only_iv = args.only_iv
    SAMPLE_CONFIG.num_epochs = args.search_epochs

    # Load once
    samples, df_meta, encoders = load_or_generate_samples()

    base_cfg = vars(copy.deepcopy(args))  # base CLI values

    # Write header once
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["param", "value"] + ["avg_loss", "f1_macro", "f1_micro", "auroc"] +
                        ["f1_class_0", "f1_class_1", "f1_class_2", "f1_class_3"] +
                        ["precision_class_0", "precision_class_1", "precision_class_2", "precision_class_3"] +
                        ["recall_class_0", "recall_class_1", "recall_class_2", "recall_class_3"])

    for param, values in grid_dict.items():
        print(f"\n=== Sweeping {param} over {values} ===")
        for val in values:
            print(f"\n→ {param} = {val}")
            args_copy = copy.deepcopy(args)
            for k, v in base_cfg.items():
                setattr(args_copy, k, v)
            setattr(args_copy, param, val)

            override_config_from_args(args_copy)
            SAMPLE_CONFIG.num_epochs = args.search_epochs

            metrics = run(samples=samples, df_meta=df_meta, encoders=encoders)

            row = [param, val] + [
                metrics["avg_loss"],
                metrics["f1_macro"],
                metrics["f1_micro"],
                metrics["auroc"]
            ] + list(metrics["f1_per_class"]) + list(metrics["precision_per_class"]) + list(metrics["recall_per_class"])

            with open(results_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"F1_macro: {metrics['f1_macro']:.4f} | AUROC: {metrics['auroc']:.4f}")

    print(f"\nLine search complete. Results saved to: {results_csv}")

if __name__ == "__main__":
    args = parse_args()
    if not args.grid_search:
        print("Missing --grid_search flag. Exiting.")
    else:
        line_search(args)
