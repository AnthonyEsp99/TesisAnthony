import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    brier_score_loss, roc_curve
)

# 1. Definir el modelo simple (coincide con los pesos)

class NeuralODE(nn.Module):
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
        return x

# -------------------------------
# 2. Utilidades (igual que antes)
# -------------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_figure_multiformat(base_path, formats=['pdf', 'svg']):
    for fmt in formats:
        plt.savefig(f"{base_path}.{fmt}", bbox_inches='tight')
        print(f"  Guardado: {base_path}.{fmt}")
    plt.close()

def compute_bias(y_true, y_prob):
    return float(np.mean(y_prob) - np.mean(y_true))

def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_low, bin_high = bin_edges[i], bin_edges[i+1]
        in_bin = (y_prob >= bin_low) & (y_prob < bin_high)
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_low) & (y_prob <= bin_high)
        if np.sum(in_bin) == 0:
            continue
        prob_true = np.mean(y_true[in_bin])
        prob_pred = np.mean(y_prob[in_bin])
        ece += (np.sum(in_bin) / len(y_true)) * np.abs(prob_pred - prob_true)
    return float(ece)

def calibration_curve_manual(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_prob, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.array([np.mean(y_prob)]), np.array([np.mean(y_true)])
    pred_means, true_means = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        pred_means.append(np.mean(y_prob[mask]))
        true_means.append(np.mean(y_true[mask]))
    return np.array(pred_means), np.array(true_means)

def plot_calibration_curve(y_true, y_prob, title, out_base):
    plt.figure(figsize=(7, 6))
    prob_pred, prob_true = calibration_curve_manual(y_true, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Neural ODE')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Probabilidad media predicha')
    plt.ylabel('Frecuencia real positiva')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_figure_multiformat(out_base)

def plot_roc_curve(y_true, y_prob, title, out_base):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'Neural ODE (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_figure_multiformat(out_base)

def plot_residual_hist(y_prob, y_true, out_base):
    residuals = y_prob - y_true
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=50, alpha=0.7, color='steelblue')
    plt.xlabel('Residuo (prob. predicha - etiqueta real)')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de residuos')
    plt.tight_layout()
    save_figure_multiformat(out_base)

def plot_noise_stability(mean_change, std_change, out_base):
    plt.figure(figsize=(6, 5))
    plt.bar(['Neural ODE'], [mean_change], yerr=[std_change], capsize=5, color='blue')
    plt.ylabel('Cambio medio absoluto en la probabilidad')
    plt.title('Estabilidad bajo ruido')
    plt.tight_layout()
    save_figure_multiformat(out_base)

# -------------------------------
# 3. Carga del modelo y predicciones
# -------------------------------
def get_input_dim_from_checkpoint(model_path):
    state = torch.load(model_path, map_location='cpu')
    return state["fc1.weight"].shape[1]

def load_model(model_path, input_dim, device):
    model = NeuralODE(input_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Modelo cargado: {model_path}, input_dim={input_dim}")
    return model

def predict_probs(model, X, device, batch_size=256):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch)
            probs_batch = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.append(probs_batch)
    return np.concatenate(probs)

def noise_stability(model, X, y, device, sigma=0.05, n_repeats=10, seed=42):
    rng = np.random.default_rng(seed)
    prob_base = predict_probs(model, X, device)
    abs_deltas = []
    for _ in range(n_repeats):
        X_noisy = X + rng.normal(0, sigma, size=X.shape)
        prob_noisy = predict_probs(model, X_noisy, device)
        delta = np.mean(np.abs(prob_noisy - prob_base))
        abs_deltas.append(delta)
    return {
        "sigma": sigma,
        "mean_abs_change": float(np.mean(abs_deltas)),
        "std_abs_change": float(np.std(abs_deltas))
    }

def make_ood_split(X, y, percentile=5, seed=42):
    low = np.percentile(X, percentile, axis=0)
    high = np.percentile(X, 100 - percentile, axis=0)
    mask_ood = np.any((X < low) | (X > high), axis=1)
    id_idx = np.where(~mask_ood)[0]
    ood_idx = np.where(mask_ood)[0]
    if len(ood_idx) > 800:
        ood_idx = np.random.choice(ood_idx, 800, replace=False)
    return id_idx, ood_idx

# -------------------------------
# 4. Función principal
# -------------------------------
def main(csv_file="dataset_tb.csv",
         model_path="tb_neural_ode_cure_best.pth",
         out_dir="Analisis_nuevo",
         tau_fixed=None,
         tau_percentile=20,
         seed=42):
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    # Determinar input_dim desde el checkpoint
    input_dim = get_input_dim_from_checkpoint(model_path)
    print(f"Dimensión de entrada detectada del modelo: {input_dim}")

    # Cargar datos
    df = pd.read_csv(csv_file)
    print(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")

    # Calcular tau
    if tau_fixed is not None:
        tau_real = tau_fixed
        print(f"Usando tau fijo = {tau_real}")
    else:
        tau_real = np.percentile(df["delta_log"], tau_percentile)
        print(f"Usando tau percentil {tau_percentile} = {tau_real:.4f}")

    # Crear etiqueta de cura
    df["cure"] = (df["delta_log"] <= tau_real).astype(float)

    # Seleccionar características: todas las columnas numéricas excepto delta_log, log_mdr, cure
    exclude = ["delta_log", "log_mdr", "cure"]
    numeric_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    print(f"Columnas numéricas disponibles: {len(numeric_cols)}")

    if len(numeric_cols) < input_dim:
        raise ValueError(f"El modelo espera {input_dim} características, pero solo hay {len(numeric_cols)} columnas numéricas disponibles.")
    
    # Tomar las primeras 'input_dim' columnas (mismo orden que en entrenamiento)
    feature_cols = numeric_cols[:input_dim]
    print(f"Usando las primeras {input_dim} columnas: {feature_cols[:5]}...")

    X = df[feature_cols].values.astype(np.float32)
    y = df["cure"].values.astype(np.float32)

    # Dividir en train/val/test (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)
    print(f"Particiones: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Cargar modelo
    model = load_model(model_path, input_dim, device)

    # Predicciones en test
    prob_test = predict_probs(model, X_test, device)
    y_pred = (prob_test >= 0.5).astype(int)

    # Métricas
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, prob_test),
        "log_loss": log_loss(y_test, prob_test),
        "brier_score": brier_score_loss(y_test, prob_test),
        "bias": compute_bias(y_test, prob_test),
        "ece": compute_ece(y_test, prob_test, n_bins=10)
    }
    print("\n=== MÉTRICAS EN TEST ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    save_json(metrics, os.path.join(out_dir, "metrics_test.json"))

    # Guardar predicciones
    pred_df = pd.DataFrame({
        "true_label": y_test,
        "prob_cura": prob_test,
        "pred_label": y_pred
    })
    pred_df.to_csv(os.path.join(out_dir, "predicciones_test.csv"), index=False)

    # Gráficos
    plot_calibration_curve(y_test, prob_test,
                           title="Curva de calibración - Test",
                           out_base=os.path.join(out_dir, "calibration_curve"))
    plot_roc_curve(y_test, prob_test,
                   title="Curva ROC - Test",
                   out_base=os.path.join(out_dir, "roc_curve"))
    plot_residual_hist(prob_test, y_test,
                       out_base=os.path.join(out_dir, "residuals_hist"))

    # Estabilidad bajo ruido
    print("\nCalculando estabilidad bajo ruido...")
    noise_res = noise_stability(model, X_test, y_test, device, sigma=0.05, n_repeats=10, seed=seed)
    save_json(noise_res, os.path.join(out_dir, "noise_stability.json"))
    plot_noise_stability(noise_res["mean_abs_change"], noise_res["std_abs_change"],
                         out_base=os.path.join(out_dir, "noise_stability_bar"))
    print(f"Cambio medio absoluto bajo ruido: {noise_res['mean_abs_change']:.5f}")

    # Evaluación OOD
    print("\nEvaluando rendimiento en datos OOD...")
    id_idx, ood_idx = make_ood_split(X_test, y_test, percentile=5, seed=seed)
    prob_id = predict_probs(model, X_test[id_idx], device)
    prob_ood = predict_probs(model, X_test[ood_idx], device)
    y_pred_id = (prob_id >= 0.5).astype(int)
    y_pred_ood = (prob_ood >= 0.5).astype(int)

    ood_metrics = {
        "ID": {
            "accuracy": accuracy_score(y_test[id_idx], y_pred_id),
            "f1_score": f1_score(y_test[id_idx], y_pred_id),
            "auc": roc_auc_score(y_test[id_idx], prob_id),
            "ece": compute_ece(y_test[id_idx], prob_id)
        },
        "OOD": {
            "accuracy": accuracy_score(y_test[ood_idx], y_pred_ood),
            "f1_score": f1_score(y_test[ood_idx], y_pred_ood),
            "auc": roc_auc_score(y_test[ood_idx], prob_ood),
            "ece": compute_ece(y_test[ood_idx], prob_ood)
        }
    }
    save_json(ood_metrics, os.path.join(out_dir, "ood_report.json"))
    print("\nRendimiento ID vs OOD:")
    print(pd.DataFrame(ood_metrics).T)

    ood_df = pd.DataFrame({
        "true_label": np.concatenate([y_test[id_idx], y_test[ood_idx]]),
        "prob_cura": np.concatenate([prob_id, prob_ood]),
        "set": ["ID"]*len(id_idx) + ["OOD"]*len(ood_idx)
    })
    ood_df.to_csv(os.path.join(out_dir, "ood_predictions.csv"), index=False)

    print(f"\nAnálisis completado. Resultados en: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="dataset_tb.csv")
    parser.add_argument("--model", type=str, default="tb_neural_ode_cure_best.pth")
    parser.add_argument("--out", type=str, default="Analisis_nuevo")
    parser.add_argument("--tau_fixed", type=float, default=None)
    parser.add_argument("--tau_percentile", type=float, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        csv_file=args.csv,
        model_path=args.model,
        out_dir=args.out,
        tau_fixed=args.tau_fixed,
        tau_percentile=args.tau_percentile,
        seed=args.seed
    )