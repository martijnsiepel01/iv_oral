# cli.py

import argparse
from types import SimpleNamespace
from config import MODEL_CONFIG, SAMPLE_CONFIG, PATHS
from config import get_sample_filenames

# Update MODEL_CONFIG defaults
MODEL_CONFIG.update({
    "model_type": "GIN",
    "emb_dim": 64,
    "lr": 0.001,
    "num_layers": 2,
    "pooling": "concat",
    "dropout": 0.0,
    "optimizer": "AdamW",
    "weight_decay": 0.0001,
    "loss": "bce",
    "binary": True,
})

def parse_args():
    parser = argparse.ArgumentParser(description="Train IV-to-Oral GNN")

    # Sample generation options
    parser.add_argument("--limit", type=int, default=SAMPLE_CONFIG.limit,
                        help="Number of patients to load")
    parser.add_argument("--only_iv", action="store_true",
                        help="Filter to patients whose first prescription is IV")
    parser.add_argument("--no_balance", action="store_true",
                        help="Do not balance switch/non-switch examples")
    parser.add_argument("--epochs", type=int, default=SAMPLE_CONFIG.num_epochs,
                        help="Number of training epochs")

    # Model options
    parser.add_argument("--model_type", choices=["GAT", "GCN", "GIN"], default=MODEL_CONFIG["model_type"])
    parser.add_argument("--emb_dim", type=int, default=MODEL_CONFIG["emb_dim"])
    parser.add_argument("--lr", type=float, default=MODEL_CONFIG["lr"])
    parser.add_argument("--num_layers", type=int, default=MODEL_CONFIG["num_layers"])
    parser.add_argument("--pooling", choices=["max", "mean", "concat"], default=MODEL_CONFIG["pooling"])
    parser.add_argument("--dropout", type=float, default=MODEL_CONFIG["dropout"])
    parser.add_argument("--optimizer", choices=["AdamW", "Adam", "RAdam"], default=MODEL_CONFIG["optimizer"])
    parser.add_argument("--weight_decay", type=float, default=MODEL_CONFIG["weight_decay"])

    # Loss function
    parser.add_argument("--loss", choices=["focal", "bce"], default=MODEL_CONFIG["loss"],
                        help="Loss function to use (focal or bce)")
    
    # Type of output
    parser.add_argument("--binary", dest="binary", action="store_true", help="Use binary classification")
    parser.add_argument("--four_way", dest="binary", action="store_false", help="Use 4-way classification")
    parser.set_defaults(binary=MODEL_CONFIG["binary"])

    # Hyperparameter search
    parser.add_argument("--grid_search", action="store_true", help="Run a predefined hyperparameter grid search")
    parser.add_argument("--search_epochs", type=int, default=10, help="Number of epochs per configuration during search")
    parser.add_argument("--param_grid", type=str, help="Path to JSON/YAML describing the hyper-parameter grid")
    
    parser.add_argument("--graph_construction_type", type=int, default=1, help="Graph construction variation type, 1-3")

    parser.add_argument("--downsample", action="store_true", help="Run downsample strategy")

    parser.add_argument("--long_iv", action="store_true", help="Predict long IV continuation (5 days)")

    return parser.parse_args()

def override_config_from_args(args):
    SAMPLE_CONFIG.limit     = args.limit
    SAMPLE_CONFIG.balanced  = not args.no_balance
    SAMPLE_CONFIG.only_iv   = args.only_iv
    SAMPLE_CONFIG.num_epochs = args.epochs
    SAMPLE_CONFIG.downsample = args.downsample

    pt_path, parquet_path = get_sample_filenames(
        args.limit,
        args.only_iv,
        not args.no_balance,
        args.graph_construction_type,
        args.long_iv,
    )
    SAMPLE_CONFIG.samples_path = pt_path
    SAMPLE_CONFIG.meta_path = parquet_path
    SAMPLE_CONFIG.graph_type = args.graph_construction_type
    SAMPLE_CONFIG.long_iv = args.long_iv

    MODEL_CONFIG.update({
        "model_type": args.model_type,
        "emb_dim": args.emb_dim,
        "lr": args.lr,
        "num_layers": args.num_layers,
        "pooling": args.pooling,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "loss": args.loss,
        "binary": True if args.long_iv else args.binary,
        "graph_construction_type": args.graph_construction_type,
        "long_iv": args.long_iv
    })

    # Return merged config (used for naming and tracking)
    return {
        **MODEL_CONFIG,
        "limit": args.limit,
        "only_iv": args.only_iv,
        "balanced": not args.no_balance,
        "epochs": args.epochs,
        "downsample": args.downsample,
        "long_iv": args.long_iv,
    }