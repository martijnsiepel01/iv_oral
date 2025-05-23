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




def get_model_name(config: dict) -> str:
    def sanitize(val): return str(val).replace('.', 'p')
    config_str = "_".join(f"{k}-{sanitize(v)}" for k, v in sorted(config.items()))
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
    
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2 if MODEL_CONFIG.get("binary", False) else 4

    # Load data
    samples, df_meta, encoders = load_or_generate_samples()
      
    ds = IVSwitchGraphDataset(samples)
    tr_idx, va_idx = train_test_split(range(len(ds)), test_size=0.2, random_state=0)
    tr_lbl = torch.stack([samples[i][1] for i in tr_idx])

    tr_dl = DataLoader(torch.utils.data.Subset(ds, tr_idx), batch_size=8, shuffle=True, collate_fn=collate_fn)
    va_dl = DataLoader(torch.utils.data.Subset(ds, va_idx), batch_size=8, shuffle=False, collate_fn=collate_fn)

    model = create_model(samples[0][0].x.shape[1], 4, MODEL_CONFIG, num_classes=num_classes).to(device)
    opt = create_optimizer(model, MODEL_CONFIG)

    # Compute class weights
    class_counts = torch.bincount(tr_lbl, minlength=num_classes)
    class_weights = torch.ones(num_classes, dtype=torch.float)

    # Avoid division by zero
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_weights[i] = 1.0 / class_counts[i].float()

    # Normalize
    class_weights = class_weights / class_weights.sum() * num_classes
    
    assert class_weights.shape[0] == num_classes, (
        f"Expected {num_classes} class weights but got {class_weights.shape[0]}"
    )

    if MODEL_CONFIG["loss"] == "focal":
        loss_fn = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model_name = get_model_name(MODEL_CONFIG)
    model_path = PATHS.models / f"{model_name}.pt"

    best_val_f1 = 0
    best_epoch = 0
    patience = 3
    no_improve_count = 0

    for epoch in range(1, SAMPLE_CONFIG.num_epochs + 1):
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
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        print("")

        if metrics["f1_macro"] > best_val_f1:
            best_val_f1 = metrics["f1_macro"]
            best_epoch = epoch
            no_improve_count = 0
            torch.save(model.state_dict(), model_path)
            print(f"✓ New best model saved to {model_path}")
        else:
            no_improve_count += 1
            print(f"No improvement. Patience: {no_improve_count}/{patience}")
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch was {best_epoch})")
                break