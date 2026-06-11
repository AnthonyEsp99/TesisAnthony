# ==================================================
# train_cure_corregido.py
# Entrenamiento de clasificador de cura (modelo simple)
# Mejoras:
#   - Pesos de clase (pos_weight)
#   - Early stopping por F1 (no por loss)
#   - Ajuste de umbral en validación
#   - Guardado del mejor modelo según F1
# ==================================================

import os
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve

# -------------------------------
# 1. Modelo simple (el que funciona)
# -------------------------------
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x  # logits

# -------------------------------
# 2. Utilidades
# -------------------------------
def compute_cure_label(df, tau_fixed=None, tau_percentile=20):
    """Crea columna 'cure' a partir de delta_log y el umbral tau."""
    if tau_fixed is not None:
        tau = tau_fixed
        print(f"Usando tau fijo = {tau}")
    else:
        tau = np.percentile(df["delta_log"], tau_percentile)
        print(f"Usando tau percentil {tau_percentile} = {tau:.4f}")
    df = df.copy()
    df["cure"] = (df["delta_log"] <= tau).astype(float)
    return df, tau

def prepare_features(df, input_dim):
    """Selecciona las primeras 'input_dim' columnas numéricas (excluyendo delta_log, log_mdr, cure)."""
    exclude = ["delta_log", "log_mdr", "cure"]
    numeric_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    if len(numeric_cols) < input_dim:
        raise ValueError(f"Se necesitan {input_dim} columnas, pero solo hay {len(numeric_cols)} numéricas.")
    feature_cols = numeric_cols[:input_dim]
    print(f"Usando {input_dim} características: {feature_cols[:5]}...")
    X = df[feature_cols].values.astype(np.float32)
    return X, feature_cols

# -------------------------------
# 3. Entrenamiento con early stopping por F1
# -------------------------------
def train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                input_dim, epochs=80, batch_size=64, lr=1e-4,
                weight_decay=1e-3, patience=20, device=None,
                save_best_path="tb_neural_ode_cure_best.pth"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Convertir a tensores y crear loaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_dataset   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val).unsqueeze(1))
    test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Modelo
    model = SimpleClassifier(input_dim).to(device)

    # Pesos de clase (pos_weight = negativos / positivos)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Early stopping por F1
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_model_state = None

    print("\n=== Inicio del entrenamiento ===")
    for epoch in range(1, epochs + 1):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validación (con ajuste de umbral)
        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                logits = model(Xb)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_probs.extend(probs)
                val_labels.extend(yb.numpy().flatten())

        # Encontrar umbral que maximiza F1 en validación
        prec, rec, thresholds = precision_recall_curve(val_labels, val_probs)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
        best_idx = np.argmax(f1_scores[:-1])  # último threshold no tiene f1 asociado
        best_thresh = thresholds[best_idx] if len(thresholds) > 0 else 0.5
        val_pred = (np.array(val_probs) >= best_thresh).astype(int)
        val_f1 = f1_score(val_labels, val_pred)
        val_acc = accuracy_score(val_labels, val_pred)
        val_auc = roc_auc_score(val_labels, val_probs)

        scheduler.step(val_f1)  # monitor F1

        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f}")
        print(f"  Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f} | Best threshold: {best_thresh:.4f}")

        # Early stopping basado en F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"  >>> Nuevo mejor modelo (F1={best_val_f1:.4f}) <<<")
        else:
            epochs_no_improve += 1
            print(f"  -> Sin mejora ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"Early stopping en época {epoch}. Mejor F1: {best_val_f1:.4f}")
                break

    # Restaurar mejor modelo
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), save_best_path)
    print(f"\nMejor modelo guardado en {save_best_path}")

    # Evaluación en test con umbral óptimo (recalculado sobre test)
    model.eval()
    test_probs = []
    test_labels = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            test_probs.extend(probs)
            test_labels.extend(yb.numpy().flatten())

    prec_t, rec_t, thresh_t = precision_recall_curve(test_labels, test_probs)
    f1_t = 2 * (prec_t * rec_t) / (prec_t + rec_t + 1e-9)
    best_idx_t = np.argmax(f1_t[:-1])
    best_thresh_test = thresh_t[best_idx_t] if len(thresh_t) > 0 else 0.5
    test_pred = (np.array(test_probs) >= best_thresh_test).astype(int)
    test_acc = accuracy_score(test_labels, test_pred)
    test_f1 = f1_score(test_labels, test_pred)
    test_auc = roc_auc_score(test_labels, test_probs)

    print("\n=== Evaluación en test ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 score: {test_f1:.4f}")
    print(f"AUC:      {test_auc:.4f}")
    print(f"Umbral óptimo en test: {best_thresh_test:.4f}")

    return model, {"best_val_f1": best_val_f1, "test_acc": test_acc, "test_f1": test_f1, "test_auc": test_auc}

# -------------------------------
# 4. Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Entrenamiento corregido para clasificador de cura")
    parser.add_argument("--csv", type=str, default="dataset_tb.csv")
    parser.add_argument("--input_dim", type=int, default=20, help="Número de características de entrada (debe coincidir con el modelo)")
    parser.add_argument("--tau_fixed", type=float, default=None, help="Umbral fijo para definir cura (sobreescribe percentil)")
    parser.add_argument("--tau_percentile", type=float, default=20, help="Percentil de delta_log para definir cura")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Semilla
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Cargar datos
    df = pd.read_csv(args.csv)
    print(f"Datos cargados: {len(df)} filas")

    # Crear etiqueta cure
    df, tau = compute_cure_label(df, args.tau_fixed, args.tau_percentile)

    # Preparar características (primeras 'input_dim' columnas numéricas)
    X, feature_cols = prepare_features(df, args.input_dim)
    y = df["cure"].values.astype(np.float32)

    # Dividir en train/val/test (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=args.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=y_temp)
    print(f"Particiones: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Proporción de cura en train: {y_train.mean():.3f}, val: {y_val.mean():.3f}, test: {y_test.mean():.3f}")

    # Entrenar
    model, metrics = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        input_dim=args.input_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience
    )

    # Guardar métricas
    with open("training_metrics_cure.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nMétricas guardadas en training_metrics_cure.json")

if __name__ == "__main__":
    main()