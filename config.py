# config.py

from pathlib import Path
from types import SimpleNamespace

# Paths
PATHS = SimpleNamespace(
    data_json=Path(r"C:/Users/Martijn/OneDrive/PhD/iv_oral_v0.1/data_raw/prescriptions/prescriptions_with_measurements_agg_1h.json"),
    base_dir=Path(r"C:/Users/Martijn/OneDrive/PhD/iv_oral_v0.1/iv_oral_gnn/samples"),
    models=Path("models")
)

# Sample generation config
SAMPLE_CONFIG = SimpleNamespace(
    limit=5000,
    balanced=True,
    only_iv=True,
    num_epochs=10
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
    "weight_decay": 0.0
}

# Derived filenames
BASE_NAME = f"samples_iv_oral_only_iv_{SAMPLE_CONFIG.limit}"
SAMPLE_CONFIG.samples_path = PATHS.base_dir / f"{BASE_NAME}.pt"
SAMPLE_CONFIG.meta_path = PATHS.base_dir / f"{BASE_NAME}.parquet"

# Create folders if needed
PATHS.base_dir.mkdir(parents=True, exist_ok=True)
PATHS.models.mkdir(parents=True, exist_ok=True)