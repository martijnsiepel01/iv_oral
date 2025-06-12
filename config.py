# config.py

from pathlib import Path
from types import SimpleNamespace

# Dynamically determine base directory inside user's OneDrive
USER_HOME = Path.home()
BASE_PROJECT_DIR = USER_HOME / "OneDrive" / "PhD" / "iv_oral_v0.1" / "iv_oral_gnn"
DATA_JSON_PATH = USER_HOME / "OneDrive" / "PhD" / "iv_oral_v0.1" / "data_raw" / "prescriptions" / "prescriptions_with_measurements_agg_1h_old.json"

# Paths
PATHS = SimpleNamespace(
    data_json=DATA_JSON_PATH,
    base_dir=BASE_PROJECT_DIR / "samples",
    models=BASE_PROJECT_DIR / "models"
)

# Sample generation config
SAMPLE_CONFIG = SimpleNamespace(
    limit=5000,
    balanced=True,
    only_iv=True,
    num_epochs=10,
    graph_type=1,
    long_iv=False
)

# Model configuration
MODEL_CONFIG = {
    "model_type": "GAT",
    "emb_dim": 32,
    "lr": 3e-4,
    "num_layers": 1,
    "pooling": "max",
    "dropout": 0.0,
    "optimizer": "AdamW",
    "weight_decay": 0.0,
    "binary": True,
    "long_iv": False
}

# Derived filenames
BASE_NAME = "samples_long_iv" if SAMPLE_CONFIG.long_iv else "samples_iv_oral"
BASE_NAME += f"_only_iv_{SAMPLE_CONFIG.limit}" if SAMPLE_CONFIG.only_iv else f"_all_{SAMPLE_CONFIG.limit}"
SAMPLE_CONFIG.samples_path = PATHS.base_dir / f"{BASE_NAME}.pt"
SAMPLE_CONFIG.meta_path = PATHS.base_dir / f"{BASE_NAME}.parquet"

# Create folders if needed
PATHS.base_dir.mkdir(parents=True, exist_ok=True)
PATHS.models.mkdir(parents=True, exist_ok=True)

# Utility function
def get_sample_filenames(limit, only_iv, balanced, graph_type, long_iv=False):
    name = "samples_long_iv" if long_iv else "samples_iv_oral"
    name += "_only_iv" if only_iv else "_all"
    name += "_balanced" if balanced else "_unbalanced"
    name += f"_{limit}"
    name += f"_graph_type_{graph_type}"
    return (
        PATHS.base_dir / f"{name}.pt",
        PATHS.base_dir / f"{name}.parquet",
    )

MISSING_VALUE = "NA"
