from model.gnn import *
from datetime import datetime
import torch
from config import MODEL_CONFIG, SAMPLE_CONFIG, PATHS

from model.dataset import IVSwitchGraphDataset, collate_fn
from model.gnn import GNNModel
from model.loss import FocalLoss
from model.train_eval import train, validate
from pipeline.sample_generation import load_or_generate_samples
from config import MODEL_CONFIG, SAMPLE_CONFIG, PATHS

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import wandb

import wandb
from graph.builder import build_pyg_graph_with_all_features_1, build_pyg_graph_with_all_features_2, build_pyg_graph_with_all_features_3


def get_model_name(config: dict) -> str:
    def sanitize(val):
        if isinstance(val, bool):
            return "T" if val else "F"
        return str(val).replace('.', 'p')

    short_keys = {
        "model_type": "mt",
        "emb_dim": "ed",
        "lr": "lr",
        "num_layers": "nl",
        "pooling": "pl",
        "dropout": "do",
        "optimizer": "opt",
        "weight_decay": "wd",
        "loss": "ls",
        "binary": "bin",
        "epochs": "ep",
        "limit": "lim",
        "only_iv": "iv",
        "balanced": "bal",
        "downsample": "ds",
        "graph_construction_type": "gt",
        "long_iv": "liv"
    }

    config_str = "_".join(f"{short_keys.get(k, k)}-{sanitize(v)}" for k, v in sorted(config.items()))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{config['model_type']}_{timestamp}_{config_str}"


def create_model(in_dim, node_type_vocab_size, config, num_classes):
    return GNNModel(
        in_dim=in_dim,
        node_type_vocab_size=node_type_vocab_size,
        model_type=config["model_type"],
        emb_dim=config["emb_dim"],
        num_layers=config["num_layers"],
        pooling=config["pooling"],
        dropout=config["dropout"],
        num_classes=num_classes
    )

def create_optimizer(model, config):
    if config["optimizer"] == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "RAdam":
        return torch.optim.RAdam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
def run(config, samples=None, df_meta=None, encoders=None):
    wandb.init(project="iv-oral-switch", name=get_model_name(config), config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2 if config.get("binary", False) else 4

    # Load data only if not already passed in
    if samples is None or df_meta is None or encoders is None:
        samples, df_meta, encoders = load_or_generate_samples()

    lbl = torch.tensor([y.item() for _, y in samples])
    print("Positives:", (lbl == 1).sum().item(), "Negatives:", (lbl == 0).sum().item())

    GRAPH_BUILDERS = {
        1: (build_pyg_graph_with_all_features_1, 4),
        2: (build_pyg_graph_with_all_features_2, 5),
        3: (build_pyg_graph_with_all_features_3, 5)
    }
    builder_fn, expected_node_type_vocab_size = GRAPH_BUILDERS[config["graph_construction_type"]]

    unique_types = set()
    for i in range(min(3, len(samples))):
        unique_types.update(samples[i][0].node_type_ids.tolist())
        x = samples[i][0].x
        print(f"x stats: min={x.min().item():.1f} max={x.max().item():.1f} mean={x.mean().item():.2f} std={x.std().item():.2f}")
    print("Unique node type IDs:", sorted(unique_types))
    print("Expected vocab size:", expected_node_type_vocab_size)

    if max(unique_types) + 1 != expected_node_type_vocab_size:
        print("[WARNING] node_type_vocab_size mismatch — check if all types are present in training data")

    model = create_model(samples[0][0].x.shape[1], expected_node_type_vocab_size, config, num_classes=num_classes).to(device)
    wandb.watch(model, log="all", log_freq=10)

    ds = IVSwitchGraphDataset(samples)
    tr_idx, va_idx = train_test_split(range(len(ds)), test_size=0.2, random_state=0)
    tr_lbl = torch.stack([samples[i][1] for i in tr_idx])

    print("Training label counts:", tr_lbl.sum(dim=0).tolist(), "| Total:", len(tr_lbl))

    tr_dl = DataLoader(torch.utils.data.Subset(ds, tr_idx), batch_size=16, shuffle=True, collate_fn=collate_fn)
    va_dl = DataLoader(torch.utils.data.Subset(ds, va_idx), batch_size=8, shuffle=False, collate_fn=collate_fn)

    opt = create_optimizer(model, config)

    num_classes = int(tr_lbl.max().item()) + 1
    class_counts = torch.bincount(tr_lbl, minlength=num_classes).float()
    class_weights = torch.where(class_counts > 0,
                                class_counts.max() / class_counts,
                                torch.zeros_like(class_counts))
    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class weights (inverse freq): {class_weights.tolist()}")

    if config["loss"] == "focal":
        loss_fn = FocalLoss(alpha=class_weights.to(device), gamma=1.0)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model_name = get_model_name(config)
    model_path = PATHS.models / f"{model_name}.pt"

    best_val_f1 = 0
    best_epoch = 0
    patience = 10
    no_improve_count = 0

    for epoch in range(1, config["epochs"] + 1):
        print(f"[Epoch {epoch}]")
        train_loss = train(model, tr_dl, opt, loss_fn, device)
        metrics = validate(model, va_dl, loss_fn, device, num_classes)

        print(f"Loss:          {metrics['avg_loss']:.4f}")
        print(f"F1 Macro:      {metrics['f1_macro']:.4f}")
        print(f"F1 Micro:      {metrics['f1_micro']:.4f}")
        print(f"AUROC:         {metrics['auroc']:.4f}")
        print(f"Per-label F1:  {metrics['f1_per_class']}")
        print(f"Precision:     {metrics['precision_per_class']}")
        print(f"Recall:        {metrics['recall_per_class']}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")

        log_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics['avg_loss'],
            "f1_macro": metrics['f1_macro'],
            "f1_micro": metrics['f1_micro'],
            "auroc": metrics['auroc'],
        }

        for i, (f1, prec, rec) in enumerate(zip(
                metrics['f1_per_class'],
                metrics['precision_per_class'],
                metrics['recall_per_class'])):
            log_data.update({
                f"class_{i}_f1": f1,
                f"class_{i}_precision": prec,
                f"class_{i}_recall": rec
            })

        wandb.log(log_data)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"[WARNING] No gradient for: {name}")

        if metrics["f1_macro"] > best_val_f1:
            best_val_f1 = metrics["f1_macro"]
            best_epoch = epoch
            no_improve_count = 0
            best_metrics = metrics
            torch.save(model.state_dict(), model_path)
            artifact = wandb.Artifact("best-model", type="model")
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)
            print(f"✓ New best model saved to {model_path}")
        else:
            no_improve_count += 1
            print(f"No improvement. Patience: {no_improve_count}/{patience}")
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch was {best_epoch})")
                break

    wandb.finish()
    return best_metrics if 'best_metrics' in locals() else metrics
