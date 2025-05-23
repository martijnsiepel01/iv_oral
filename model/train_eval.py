from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score
import numpy as np


def train(model, dataloader, optimizer, loss_fn, device):
    print("Start training")
    model.train()
    all_preds, all_labels = [], []
    for g, y in tqdm(dataloader, desc="Training"):
        g, y = g.to(device), y.to(device)
        if torch.isnan(g.x).any():
            print("[FATAL] NaNs in graph.x")
            print("Offending x:", g.x)
            print("Batch indices:", g.batch)
            print("Node type IDs:", g.node_type_ids)
            raise ValueError("NaNs in input data.x")
        if g.node_type_ids.min() < 0 or g.node_type_ids.max() >= model.node_type_emb.num_embeddings:
            print("[FATAL] Invalid node_type_ids:", g.node_type_ids)
            raise ValueError("Invalid node_type_ids for embedding")
        pred = model(g)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_preds.append(torch.softmax(pred, dim=1).cpu().detach())
        all_labels.append(y.cpu())
    return loss.item()

def validate(model, dataloader, loss_fn, device, num_classes) -> tuple:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for g, y in tqdm(dataloader, desc="Validating"):
            g, y = g.to(device), y.to(device)
            logits = model(g)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            all_logits.append(torch.softmax(logits, dim=1).cpu())
            all_labels.append(y.cpu())

    probs  = torch.cat(all_logits)             # [N,4]
    y_true = torch.cat(all_labels).numpy()     # [N]
    y_pred = probs.argmax(dim=1).numpy()       # [N]

    metrics = {
        "avg_loss":   total_loss / len(dataloader),
        "f1_macro":   f1_score(y_true, y_pred, average="macro"),
        "f1_micro":   f1_score(y_true, y_pred, average="micro"),
        "f1_per_class": f1_score(y_true, y_pred, average=None),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0),
        "auroc": roc_auc_score(
            np.eye(num_classes)[y_true], probs.numpy(), multi_class="ovr" if num_classes > 2 else "raise"
        ),
        "confusion_matrix":   confusion_matrix(y_true, y_pred),
        "report":     classification_report(y_true, y_pred, digits=3)
    }

    return metrics