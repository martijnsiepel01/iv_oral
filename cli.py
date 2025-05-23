# cli.py

import argparse
from types import SimpleNamespace
from config import MODEL_CONFIG, SAMPLE_CONFIG, PATHS

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
    parser.add_argument("--loss", choices=["focal", "bce"], default=MODEL_CONFIG.get("loss", "focal"),
                        help="Loss function to use (focal or bce)")
    
    # Type of output
    parser.add_argument("--binary", action="store_true", help="Use binary classification")
    
    # Hyperparameter search
    parser.add_argument("--grid_search", action="store_true", help="Run a predefined hyperparameter grid search")
    parser.add_argument("--search_epochs", type=int, default=10, help="Number of epochs per configuration during search")

    return parser.parse_args()

def override_config_from_args(args):
    # Update SAMPLE_CONFIG
    SAMPLE_CONFIG.limit = args.limit
    SAMPLE_CONFIG.balanced = not args.no_balance
    SAMPLE_CONFIG.only_iv = args.only_iv
    SAMPLE_CONFIG.num_epochs = args.epochs

    # Update paths
    from config import BASE_NAME
    SAMPLE_CONFIG.samples_path = PATHS.base_dir / f"samples_iv_oral_only_iv_{args.limit}.pt"
    SAMPLE_CONFIG.meta_path = PATHS.base_dir / f"samples_iv_oral_only_iv_{args.limit}.parquet"

    # Update MODEL_CONFIG
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
    })
    
    MODEL_CONFIG["binary"] = args.binary