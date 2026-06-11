import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # cargar el scaler de estáticas

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import torch
import torch.nn as nn

from neural_ode_model import TB_NeuralODE

# Utilidades

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def z_to_real(z, mu, sd):
    return z * sd + mu

def real_to_z(x, mu, sd):
    return (x - mu) / (sd + 1e-12)

def code_to_onehot(code_seq):
    
    # Convierte una lista de códigos 0..15 a un array one-hot de 4 bits.
    
    T = len(code_seq)
    seq = np.zeros((T, 4), dtype=np.float32)
    for t, c in enumerate(code_seq):
        c = int(c) & 0xF
        seq[t, 0] = 1.0 if (c & 1) else 0.0
        seq[t, 1] = 1.0 if (c & 2) else 0.0
        seq[t, 2] = 1.0 if (c & 4) else 0.0
        seq[t, 3] = 1.0 if (c & 8) else 0.0
    return seq

def prepare_input_from_case(beta_s, beta_r, log_q_H, log_q_R, log_q_S, log_q_Z,
                            s0, r0, delta_s, delta_r, alpha_H, alpha_R, alpha_S, alpha_Z,
                            treat_code, T=180, device="cpu", scaler=None):
    
    static_vec = np.array([beta_s, beta_r, log_q_H, log_q_R, log_q_S, log_q_Z,
                           s0, r0, delta_s, delta_r, alpha_H, alpha_R, alpha_S, alpha_Z],
                          dtype=np.float32)
    if scaler is not None:
        static_vec = scaler.transform(static_vec.reshape(1, -1)).flatten()
    static = torch.tensor(static_vec.reshape(1, -1), dtype=torch.float32, device=device)

    seq_np = code_to_onehot(treat_code[:T])
    seq = torch.tensor(seq_np[None, :, :], dtype=torch.float32, device=device)
    return static, seq

def safe_model_forward(model, static, seq, dt=0.1, method='dopri5'):
    # Intenta forward con dt y method (para Neural ODE); si falla, intenta solo con dt, y luego sin argumentos extra.
    try:
        return model(static, seq, dt=dt, method=method)
    except TypeError:
        try:
            return model(static, seq, dt=dt)
        except TypeError:
            return model(static, seq)

# Cargador de LSTM (arquitectura con 14 estáticas, sin normalización)

class LSTMBaselineCheckpoint(nn.Module):
    
    # Copia fiel de TB_LSTM_Baseline con static_dim=14.
    
    def __init__(self, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.static_net = nn.Sequential(
            nn.Linear(14, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, static, seq):
        s = self.static_net(static)
        out, _ = self.lstm(seq)
        hT = out[:, -1, :]
        x = torch.cat([s, hT], dim=1)
        y = self.head(x)
        return y[:, 0], y[:, 1]

def maybe_load_lstm(device, path="tb_lstm_best.pth"):
    """Carga el modelo LSTM si existe y la arquitectura coincide."""
    if not os.path.exists(path):
        print("No se encontró tb_lstm_best.pth. Se continúa sin LSTM.")
        return None

    state = torch.load(path, map_location=device)
    model = LSTMBaselineCheckpoint(hidden=64, num_layers=2, dropout=0.2).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print("No se pudo cargar el LSTM (arquitectura no coincide). Se continúa sin LSTM.")
        print("Detalle:", str(e)[:500], "...")
        return None

    model.eval()
    print("LSTM cargado:", path)
    return model


# Métricas y gráficos 

def compute_bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))

def calibration_curve(y_true, y_pred, n_bins=10):
    """Calcula la curva de calibración (media predicha vs media real por bins)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_pred, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.array([np.mean(y_pred)]), np.array([np.mean(y_true)]), np.array([len(y_true)])

    pred_means, true_means, counts = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() == 0:
            continue
        pred_means.append(np.mean(y_pred[mask]))
        true_means.append(np.mean(y_true[mask]))
        counts.append(int(mask.sum()))
    return np.array(pred_means), np.array(true_means), np.array(counts)

def save_figure_multiformat(base_path, formats=['pdf', 'svg']):
    """Guarda la figura actual en los formatos especificados."""
    for fmt in formats:
        path = f"{base_path}.{fmt}"
        plt.savefig(path, bbox_inches='tight')
        print(f"  Guardado: {path}")

def plot_scatter(y_true, y_pred_ode, y_pred_lstm, title, out_base):
    """Gráfico de dispersión: verdadero vs predicho."""
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred_ode, alpha=0.35, label="Neural ODE")
    if y_pred_lstm is not None:
        plt.scatter(y_true, y_pred_lstm, alpha=0.35, label="LSTM")
    mn = float(min(np.min(y_true), np.min(y_pred_ode)))
    mx = float(max(np.max(y_true), np.max(y_pred_ode)))
    if y_pred_lstm is not None:
        mn = float(min(mn, np.min(y_pred_lstm)))
        mx = float(max(mx, np.max(y_pred_lstm)))
    plt.plot([mn, mx], [mn, mx], linewidth=2, label="Ideal")
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_figure_multiformat(out_base)
    plt.close()

def plot_calibration(y_true, y_pred_ode, y_pred_lstm, title, out_base):
    """Curva de calibración (media predicha vs media real por bins)."""
    plt.figure(figsize=(7, 6))
    p_m, t_m, _ = calibration_curve(y_true, y_pred_ode, n_bins=10)
    plt.plot(p_m, t_m, marker="o", linewidth=2, label="Neural ODE")
    if y_pred_lstm is not None:
        p_m2, t_m2, _ = calibration_curve(y_true, y_pred_lstm, n_bins=10)
        plt.plot(p_m2, t_m2, marker="o", linewidth=2, label="LSTM")
    mn = float(min(np.min(y_true), np.min(y_pred_ode)))
    mx = float(max(np.max(y_true), np.max(y_pred_ode)))
    if y_pred_lstm is not None:
        mn = float(min(mn, np.min(y_pred_lstm)))
        mx = float(max(mx, np.max(y_pred_lstm)))
    plt.plot([mn, mx], [mn, mx], linewidth=2, label="Ideal")
    plt.xlabel("Media predicha (bin)")
    plt.ylabel("Media real (bin)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_figure_multiformat(out_base)
    plt.close()

def plot_residual_hist(residuals_ode, residuals_lstm, title, out_base):
    """Histograma de residuos."""
    plt.figure(figsize=(7, 6))
    plt.hist(residuals_ode, bins=50, alpha=0.6, label="Neural ODE")
    if residuals_lstm is not None:
        plt.hist(residuals_lstm, bins=50, alpha=0.6, label="LSTM")
    plt.xlabel("Residuo = predicho - real")
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_figure_multiformat(out_base)
    plt.close()

def plot_noise_report(noise_dict, out_base):
    
    #Gráfico de barras para la estabilidad bajo ruido.
    
    labels = ["Neural ODE"]
    delta_vals = [noise_dict["mean_abs_change_delta_log_ode"]]
    mdr_vals   = [noise_dict["mean_abs_change_log_mdr_ode"]]
    if "mean_abs_change_delta_log_lstm" in noise_dict:
        labels.append("LSTM")
        delta_vals.append(noise_dict["mean_abs_change_delta_log_lstm"])
        mdr_vals.append(noise_dict["mean_abs_change_log_mdr_lstm"])

    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.15, delta_vals, width=0.3, label="delta_log")
    plt.bar(x + 0.15, mdr_vals, width=0.3, label="log_mdr")
    plt.xticks(x, labels)
    plt.ylabel("Cambio medio absoluto en la predicción bajo ruido")
    plt.title("Estabilidad bajo ruido (menor es mejor)")
    plt.legend()
    plt.tight_layout()
    save_figure_multiformat(out_base)
    plt.close()

# Ayudantes de inferencia (modificados para usar scaler)

def predict_from_df(model, lstm, device, stats, df, dt=0.1, method='dopri5', T=180, scaler=None):
    #Genera predicciones para todo el DataFrame.
    #scaler: se aplica a las estáticas solo para el modelo ODE (no para LSTM).
    y_true_delta = df["delta_log"].to_numpy().astype(np.float64)
    y_true_mdr   = df["log_mdr"].to_numpy().astype(np.float64)

    pred_delta_ode, pred_mdr_ode = [], []
    pred_delta_lstm, pred_mdr_lstm = [], []

    for _, row in df.iterrows():
        treat = [int(row[f"treat_{k}"]) for k in range(T)]
        # Para ODE: con scaler
        static_ode, seq = prepare_input_from_case(
            row["beta_s"], row["beta_r"],
            row["log_q_H"], row["log_q_R"], row["log_q_S"], row["log_q_Z"],
            row["s0"], row["r0"],
            row["delta_s"], row["delta_r"],
            row["alpha_H"], row["alpha_R"], row["alpha_S"], row["alpha_Z"],
            treat, T=T, device=device, scaler=scaler
        )
        # Para LSTM: sin scaler
        static_lstm, _ = prepare_input_from_case(
            row["beta_s"], row["beta_r"],
            row["log_q_H"], row["log_q_R"], row["log_q_S"], row["log_q_Z"],
            row["s0"], row["r0"],
            row["delta_s"], row["delta_r"],
            row["alpha_H"], row["alpha_R"], row["alpha_S"], row["alpha_Z"],
            treat, T=T, device=device, scaler=None
        )

        model.eval()
        with torch.no_grad():
            pDz, pMz = safe_model_forward(model, static_ode, seq, dt=dt, method=method)
        pD = z_to_real(float(pDz.item()), stats["mu_delta"], stats["sd_delta"])
        pM = z_to_real(float(pMz.item()), stats["mu_mdr"],   stats["sd_mdr"])
        pred_delta_ode.append(pD)
        pred_mdr_ode.append(pM)

        if lstm is not None:
            lstm.eval()
            with torch.no_grad():
                pDz2, pMz2 = lstm(static_lstm, seq)
            pD2 = z_to_real(float(pDz2.item()), stats["mu_delta"], stats["sd_delta"])
            pM2 = z_to_real(float(pMz2.item()), stats["mu_mdr"],   stats["sd_mdr"])
            pred_delta_lstm.append(pD2)
            pred_mdr_lstm.append(pM2)

    pred_delta_ode = np.array(pred_delta_ode, dtype=np.float64)
    pred_mdr_ode   = np.array(pred_mdr_ode, dtype=np.float64)

    if lstm is not None:
        pred_delta_lstm = np.array(pred_delta_lstm, dtype=np.float64)
        pred_mdr_lstm   = np.array(pred_mdr_lstm, dtype=np.float64)
    else:
        pred_delta_lstm = pred_mdr_lstm = None

    return (y_true_delta, y_true_mdr,
            pred_delta_ode, pred_mdr_ode,
            pred_delta_lstm, pred_mdr_lstm)

# Estabilidad bajo ruido (modificado para usar scaler)

def noise_stability(model, lstm, device, stats, df_base, dt=0.1, method='dopri5', T=180,
                    static_sigma=0.03, treat_flip_p=0.03, n_repeats=10, seed=0, scaler=None):
    # Evalúa la sensibilidad del modelo añadiendo ruido a las estáticas y al tratamiento."""
    rng = np.random.default_rng(seed)
    df = df_base.copy().reset_index(drop=True)
    df = df.sample(n=min(300, len(df)), random_state=seed).reset_index(drop=True)

    deltas_ode, mdrs_ode = [], []
    deltas_lstm, mdrs_lstm = [], []

    # predicciones base
    yD_true, yM_true, pD0, pM0, pDL0, pML0 = predict_from_df(
        model, lstm, device, stats, df, dt=dt, method=method, T=T, scaler=scaler
    )

    for rep in range(n_repeats):
        pD_rep, pM_rep = [], []
        pDL_rep, pML_rep = [], []

        for _, row in df.iterrows():
            # Tratamiento original
            treat = np.array([int(row[f"treat_{k}"]) for k in range(T)], dtype=int)
            # Flip aleatorio en algunos días
            mask = rng.random(T) < treat_flip_p
            for k in np.where(mask)[0]:
                treat[k] = rng.integers(0, 16)

            # Variables estáticas originales
            static_vec = np.array([
                row["beta_s"], row["beta_r"],
                row["log_q_H"], row["log_q_R"], row["log_q_S"], row["log_q_Z"],
                row["s0"], row["r0"],
                row["delta_s"], row["delta_r"],
                row["alpha_H"], row["alpha_R"], row["alpha_S"], row["alpha_Z"]
            ], dtype=np.float64)
            # Ruido multiplicativo
            noise = 1.0 + rng.normal(0.0, static_sigma, size=static_vec.shape)
            static_noisy = static_vec * noise

            # Para ODE: aplicar scaler después del ruido
            static_ode = static_noisy.copy()
            if scaler is not None:
                static_ode = scaler.transform(static_ode.reshape(1, -1)).flatten()
            static_ode_t = torch.tensor(static_ode.reshape(1, -1), dtype=torch.float32, device=device)

            # Para LSTM: sin scaler
            static_lstm_t = torch.tensor(static_noisy.reshape(1, -1), dtype=torch.float32, device=device)

            seq_np = code_to_onehot(treat.tolist()[:T])
            seq_t = torch.tensor(seq_np[None, :, :], dtype=torch.float32, device=device)

            with torch.no_grad():
                pDz, pMz = safe_model_forward(model, static_ode_t, seq_t, dt=dt, method=method)
            pD = z_to_real(float(pDz.item()), stats["mu_delta"], stats["sd_delta"])
            pM = z_to_real(float(pMz.item()), stats["mu_mdr"],   stats["sd_mdr"])
            pD_rep.append(pD)
            pM_rep.append(pM)

            if lstm is not None:
                with torch.no_grad():
                    pDz2, pMz2 = lstm(static_lstm_t, seq_t)
                pD2 = z_to_real(float(pDz2.item()), stats["mu_delta"], stats["sd_delta"])
                pM2 = z_to_real(float(pMz2.item()), stats["mu_mdr"],   stats["sd_mdr"])
                pDL_rep.append(pD2)
                pML_rep.append(pM2)

        pD_rep = np.array(pD_rep)
        pM_rep = np.array(pM_rep)
        deltas_ode.append(np.mean(np.abs(pD_rep - pD0)))
        mdrs_ode.append(np.mean(np.abs(pM_rep - pM0)))

        if lstm is not None:
            pDL_rep = np.array(pDL_rep)
            pML_rep = np.array(pML_rep)
            deltas_lstm.append(np.mean(np.abs(pDL_rep - pDL0)))
            mdrs_lstm.append(np.mean(np.abs(pML_rep - pML0)))

    out = {
        "static_sigma": static_sigma,
        "treat_flip_p": treat_flip_p,
        "n_repeats": n_repeats,
        "mean_abs_change_delta_log_ode": float(np.mean(deltas_ode)),
        "mean_abs_change_log_mdr_ode": float(np.mean(mdrs_ode)),
    }
    if lstm is not None:
        out.update({
            "mean_abs_change_delta_log_lstm": float(np.mean(deltas_lstm)),
            "mean_abs_change_log_mdr_lstm": float(np.mean(mdrs_lstm)),
        })
    return out

def make_ood_split(df, seed=0):
    # Crea particiones ID (dentro de distribución) y OOD (fuera de distribución) basadas en beta_s y beta_r.
    df = df.copy().reset_index(drop=True)
    q1_bs, q9_bs = df["beta_s"].quantile(0.10), df["beta_s"].quantile(0.90)
    q1_br, q9_br = df["beta_r"].quantile(0.10), df["beta_r"].quantile(0.90)
    mask = (df["beta_s"] <= q1_bs) | (df["beta_s"] >= q9_bs) | (df["beta_r"] <= q1_br) | (df["beta_r"] >= q9_br)
    df_ood = df[mask].sample(n=min(800, mask.sum()), random_state=seed)
    df_id  = df[~mask].sample(n=min(800, (~mask).sum()), random_state=seed)
    return df_id.reset_index(drop=True), df_ood.reset_index(drop=True)

def compute_regression_report(y_true, y_pred):
    
    #Calcula MAE, MSE, R² y sesgo.
    return {
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "mse": float(np.mean((y_pred - y_true) ** 2)),
        "r2":  float(r2_score(y_true, y_pred)),
        "bias": float(compute_bias(y_true, y_pred)),
    }

# Función principal

def main(
    csv="dataset_tb.csv",
    stats_json="tb_target_stats.json",
    ode_weights="tb_neural_ode_best.pth",
    lstm_weights="tb_lstm_best.pth",
    static_scaler_path="static_scaler.pkl",
    out_dir="robustness_outputs",
    seed=42,
    dt=0.1,
    method='dopri5',
    T=180,
    latent_dim=256
):
    ensure_dir(out_dir)

    print("Cargando dataset:", csv)
    df = pd.read_csv(csv)

    treat_cols = [c for c in df.columns if c.startswith("treat_")]
    if len(treat_cols) < T:
        raise ValueError(f"Se necesitan treat_0..treat_{T-1}. Encontradas {len(treat_cols)}.")

    required_static = [
        "beta_s", "beta_r", "log_q_H", "log_q_R", "log_q_S", "log_q_Z",
        "s0", "r0", "delta_s", "delta_r", "alpha_H", "alpha_R", "alpha_S", "alpha_Z"
    ]
    for col in required_static:
        if col not in df.columns:
            raise ValueError(f"Falta la columna en el CSV: {col}")

    stats = load_json(stats_json)

    # Cargar scaler de estáticas (para Neural ODE) si existe
    scaler = None
    if os.path.exists(static_scaler_path):
        scaler = joblib.load(static_scaler_path)
        print("Scaler cargado:", static_scaler_path)
    else:
        print("No se encontró static_scaler.pkl. Se evaluará la ODE sin normalización (resultados erróneos).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    # Cargar Neural ODE con latent_dim correcto
    ode = TB_NeuralODE(latent_dim=latent_dim, static_dim=14, control_dim=4).to(device)
    state_dict = torch.load(ode_weights, map_location=device)
    ode.load_state_dict(state_dict)
    ode.eval()
    print("Neural ODE cargada:", ode_weights)

    # Cargar LSTM (opcional)
    lstm = maybe_load_lstm(device, path=lstm_weights)

    # División 70/15/15
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=seed)

    # Predicciones
    print("Prediciendo en VAL...")
    yD_val, yM_val, pD_val_ode, pM_val_ode, pD_val_lstm, pM_val_lstm = predict_from_df(
        ode, lstm, device, stats, val_df, dt=dt, method=method, T=T, scaler=scaler
    )

    print("Prediciendo en TEST...")
    yD_test, yM_test, pD_test_ode, pM_test_ode, pD_test_lstm, pM_test_lstm = predict_from_df(
        ode, lstm, device, stats, test_df, dt=dt, method=method, T=T, scaler=scaler
    )

    # Resumen de métricas
    summary = {
        "VAL": {
            "NeuralODE_delta_log": compute_regression_report(yD_val, pD_val_ode),
            "NeuralODE_log_mdr":   compute_regression_report(yM_val, pM_val_ode),
        },
        "TEST": {
            "NeuralODE_delta_log": compute_regression_report(yD_test, pD_test_ode),
            "NeuralODE_log_mdr":   compute_regression_report(yM_test, pM_test_ode),
        }
    }

    if lstm is not None:
        summary["VAL"].update({
            "LSTM_delta_log": compute_regression_report(yD_val, pD_val_lstm),
            "LSTM_log_mdr":   compute_regression_report(yM_val, pM_val_lstm),
        })
        summary["TEST"].update({
            "LSTM_delta_log": compute_regression_report(yD_test, pD_test_lstm),
            "LSTM_log_mdr":   compute_regression_report(yM_test, pM_test_lstm),
        })

    save_json(summary, os.path.join(out_dir, "metrics_summary.json"))
    print("Guardado metrics_summary.json")

    # Guardar predicciones en CSV
    preds_val = pd.DataFrame({
        "delta_log_true": yD_val,
        "log_mdr_true": yM_val,
        "delta_log_pred_ode": pD_val_ode,
        "log_mdr_pred_ode": pM_val_ode,
    })
    if lstm is not None:
        preds_val["delta_log_pred_lstm"] = pD_val_lstm
        preds_val["log_mdr_pred_lstm"]   = pM_val_lstm

    preds_test = pd.DataFrame({
        "delta_log_true": yD_test,
        "log_mdr_true": yM_test,
        "delta_log_pred_ode": pD_test_ode,
        "log_mdr_pred_ode": pM_test_ode,
    })
    if lstm is not None:
        preds_test["delta_log_pred_lstm"] = pD_test_lstm
        preds_test["log_mdr_pred_lstm"]   = pM_test_lstm

    preds_val.to_csv(os.path.join(out_dir, "preds_val.csv"), index=False)
    preds_test.to_csv(os.path.join(out_dir, "preds_test.csv"), index=False)
    print("Guardado preds_val.csv y preds_test.csv")

    # Gráficos
    plot_scatter(
        yD_test, pD_test_ode, pD_test_lstm,
        title="Valor real vs predicho: delta_log (TEST)",
        out_base=os.path.join(out_dir, "scatter_delta_log")
    )
    plot_scatter(
        yM_test, pM_test_ode, pM_test_lstm,
        title="Valor real vs predicho: log_mdr (TEST)",
        out_base=os.path.join(out_dir, "scatter_log_mdr")
    )
    plot_calibration(
        yD_test, pD_test_ode, pD_test_lstm,
        title="Curva de calibración: delta_log (TEST)",
        out_base=os.path.join(out_dir, "calibration_delta_log")
    )
    plot_calibration(
        yM_test, pM_test_ode, pM_test_lstm,
        title="Curva de calibración: log_mdr (TEST)",
        out_base=os.path.join(out_dir, "calibration_log_mdr")
    )
    res_ode = pM_test_ode - yM_test
    res_lstm = None if lstm is None else (pM_test_lstm - yM_test)
    plot_residual_hist(
        res_ode, res_lstm,
        title="Histograma de residuos: log_mdr (TEST)",
        out_base=os.path.join(out_dir, "residuals_log_mdr")
    )

    # Estabilidad bajo ruido
    print("Calculando estabilidad bajo ruido...")
    noise_report = noise_stability(
        ode, lstm, device, stats, val_df,
        dt=dt, method=method, T=T, static_sigma=0.03, treat_flip_p=0.03, n_repeats=10, seed=seed,
        scaler=scaler
    )
    save_json(noise_report, os.path.join(out_dir, "noise_report.json"))
    plot_noise_report(noise_report, os.path.join(out_dir, "noise_stability"))
    print("Guardado noise_report.json y gráficas de estabilidad")

    # Evaluación OOD
    print("Ejecutando evaluación OOD...")
    df_id, df_ood = make_ood_split(test_df, seed=seed)

    yD_id, yM_id, pD_id_ode, pM_id_ode, pD_id_lstm, pM_id_lstm = predict_from_df(
        ode, lstm, device, stats, df_id, dt=dt, method=method, T=T, scaler=scaler
    )
    yD_ood, yM_ood, pD_ood_ode, pM_ood_ode, pD_ood_lstm, pM_ood_lstm = predict_from_df(
        ode, lstm, device, stats, df_ood, dt=dt, method=method, T=T, scaler=scaler
    )

    ood_report = {
        "ID": {
            "NeuralODE_delta_log": compute_regression_report(yD_id, pD_id_ode),
            "NeuralODE_log_mdr":   compute_regression_report(yM_id, pM_id_ode),
        },
        "OOD": {
            "NeuralODE_delta_log": compute_regression_report(yD_ood, pD_ood_ode),
            "NeuralODE_log_mdr":   compute_regression_report(yM_ood, pM_ood_ode),
        }
    }
    if lstm is not None:
        ood_report["ID"].update({
            "LSTM_delta_log": compute_regression_report(yD_id, pD_id_lstm),
            "LSTM_log_mdr":   compute_regression_report(yM_id, pM_id_lstm),
        })
        ood_report["OOD"].update({
            "LSTM_delta_log": compute_regression_report(yD_ood, pD_ood_lstm),
            "LSTM_log_mdr":   compute_regression_report(yM_ood, pM_ood_lstm),
        })

    save_json(ood_report, os.path.join(out_dir, "ood_report.json"))
    print("Guardado ood_report.json")

    print("\nLISTO")
    print("Revise la carpeta:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis de robustez para modelos de tuberculosis")
    parser.add_argument("--csv", default="dataset_tb.csv")
    parser.add_argument("--stats", default="tb_target_stats.json")
    parser.add_argument("--ode", default="tb_neural_ode_best.pth")
    parser.add_argument("--lstm", default="tb_lstm_best.pth")
    parser.add_argument("--static_scaler", default="static_scaler.pkl", help="Archivo del scaler de estáticas (generado por el entrenamiento ODE)")
    parser.add_argument("--out", default="robustness_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--method", type=str, default="dopri5", choices=["euler", "rk4", "dopri5"])
    parser.add_argument("--T", type=int, default=180)
    parser.add_argument("--latent_dim", type=int, default=256, help="Debe coincidir con el entrenado")
    args = parser.parse_args()

    main(
        csv=args.csv,
        stats_json=args.stats,
        ode_weights=args.ode,
        lstm_weights=args.lstm,
        static_scaler_path=args.static_scaler,
        out_dir=args.out,
        seed=args.seed,
        dt=args.dt,
        method=args.method,
        T=args.T,
        latent_dim=args.latent_dim
    )