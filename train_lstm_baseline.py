import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Dataset y modelo (igual que antes)

class TBDatasetLSTM(Dataset):
    def __init__(self, df, seq_length=180):
        self.df = df.reset_index(drop=True)
        self.seq_length = seq_length
        self.treat_cols = [f"treat_{d}" for d in range(seq_length)]

        missing = [c for c in self.treat_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Faltan columnas treat_: {missing[:5]}...")

        self.static_cols = [
            "beta_s", "beta_r", "log_q_H", "log_q_R", "log_q_S", "log_q_Z",
            "s0", "r0", "delta_s", "delta_r", "alpha_H", "alpha_R", "alpha_S", "alpha_Z"
        ]
        for c in self.static_cols:
            if c not in self.df.columns:
                raise ValueError(f"Falta columna requerida: {c}")

        need = self.static_cols + ["delta_log_z", "log_mdr_z"]
        for c in need:
            if c not in self.df.columns:
                raise ValueError(f"Falta columna requerida: {c}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        static = torch.tensor([row[col] for col in self.static_cols], dtype=torch.float32)

        seq = torch.zeros(self.seq_length, 4, dtype=torch.float32)
        for i, c in enumerate(self.treat_cols):
            code = int(row[c])
            seq[i, 0] = 1.0 if (code & 1) else 0.0
            seq[i, 1] = 1.0 if (code & 2) else 0.0
            seq[i, 2] = 1.0 if (code & 4) else 0.0
            seq[i, 3] = 1.0 if (code & 8) else 0.0

        y_delta_z = torch.tensor(row["delta_log_z"], dtype=torch.float32)
        y_mdr_z   = torch.tensor(row["log_mdr_z"], dtype=torch.float32)

        return static, seq, y_delta_z, y_mdr_z


class TB_LSTM_Baseline(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.static_net = nn.Sequential(
            nn.Linear(14, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
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


def preparar_datos(csv="dataset_tb.csv", stats_path="tb_target_stats.json", seed=42):
    """Carga, normaliza objetivos y divide en train/val/test. No mide tiempos."""
    df = pd.read_csv(csv)
    with open(stats_path, "r") as f:
        stats = json.load(f)

    df["delta_log_z"] = (df["delta_log"] - stats["mu_delta"]) / stats["sd_delta"]
    df["log_mdr_z"]   = (df["log_mdr"]   - stats["mu_mdr"])   / stats["sd_mdr"]

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=seed)

    print(f"Dataset: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df, stats


def evaluate_mae_real(model, loader, device, stats):
    model.eval()
    mae_delta_z, mae_mdr_z, n = 0.0, 0.0, 0
    with torch.no_grad():
        for static, seq, y_delta_z, y_mdr_z in loader:
            static, seq, y_delta_z, y_mdr_z = static.to(device), seq.to(device), y_delta_z.to(device), y_mdr_z.to(device)
            p_delta_z, p_mdr_z = model(static, seq)
            mae_delta_z += torch.abs(p_delta_z - y_delta_z).sum().item()
            mae_mdr_z   += torch.abs(p_mdr_z   - y_mdr_z).sum().item()
            n += y_delta_z.size(0)
    mae_delta_real = (mae_delta_z / n) * stats["sd_delta"]
    mae_mdr_real   = (mae_mdr_z   / n) * stats["sd_mdr"]
    return mae_delta_real, mae_mdr_real


def train_lstm(train_df, val_df, stats,
               epochs=80, batch_size=64, lr=5e-4, weight_decay=1e-4,
               patience=20, w_delta=1.0, w_mdr=1.2, seq_length=180):
    """Entrena el LSTM y devuelve el modelo (mejor), el dispositivo y un diccionario con tiempos internos."""
    tiempos = {}

    train_ds = TBDatasetLSTM(train_df, seq_length=seq_length)
    val_ds   = TBDatasetLSTM(val_df, seq_length=seq_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    model = TB_LSTM_Baseline(hidden=64, layers=2, dropout=0.2).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss(beta=1.0)

    best_val = float('inf')
    wait = 0

    train_losses = []
    val_losses = []
    val_mae_delta = []
    val_mae_mdr = []

    total_start_time = time.time()
    epoch_times = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        for static, seq, y_delta_z, y_mdr_z in train_loader:
            static, seq, y_delta_z, y_mdr_z = static.to(device), seq.to(device), y_delta_z.to(device), y_mdr_z.to(device)
            opt.zero_grad()
            p_delta_z, p_mdr_z = model(static, seq)
            loss_delta = mse(p_delta_z, y_delta_z)
            loss_mdr = huber(p_mdr_z, y_mdr_z)
            loss = w_delta * loss_delta + w_mdr * loss_mdr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for static, seq, y_delta_z, y_mdr_z in val_loader:
                static, seq, y_delta_z, y_mdr_z = static.to(device), seq.to(device), y_delta_z.to(device), y_mdr_z.to(device)
                p_delta_z, p_mdr_z = model(static, seq)
                loss_delta = mse(p_delta_z, y_delta_z)
                loss_mdr = huber(p_mdr_z, y_mdr_z)
                loss = w_delta * loss_delta + w_mdr * loss_mdr
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        mae_delta_real, mae_mdr_real = evaluate_mae_real(model, val_loader, device, stats)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mae_delta.append(mae_delta_real)
        val_mae_mdr.append(mae_mdr_real)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch}/{epochs} ({epoch_time:.2f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val MAE delta_log (real): {mae_delta_real:.4f}")
        print(f"  Val MAE log_mdr   (real): {mae_mdr_real:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), "tb_lstm_best.pth")
            print("  -> Nuevo mejor modelo guardado.")
        else:
            wait += 1
            print(f"  -> Sin mejora ({wait}/{patience})")
            if wait >= patience:
                print("Early stopping activado.")
                break

    # Guardar el último modelo (por si acaso)
    torch.save(model.state_dict(), "tb_lstm_last.pth")

    training_total = time.time() - total_start_time
    epochs_completed = len(epoch_times)
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0

    tiempos["entrenamiento_total"] = training_total
    tiempos["epocas_completadas"] = epochs_completed
    tiempos["tiempo_promedio_por_epoca"] = avg_epoch_time

    print(f"\nGuardado como tb_lstm_best.pth (mejor) y tb_lstm_last.pth (último)")
    print(f"Mejor val loss: {best_val:.4f}")
    print(f"Tiempo total de entrenamiento: {training_total:.2f} segundos")

    # Generar gráficas (igual que antes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label='Entrenamiento', linewidth=2)
    ax1.plot(val_losses, label='Validación', linewidth=2)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida (MSE + Huber)')
    ax1.set_title('Curvas de pérdida - LSTM')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_mae_delta, label='MAE Δlog (real)', linewidth=2)
    ax2.plot(val_mae_mdr, label='MAE log MDR (real)', linewidth=2)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('MAE')
    ax2.set_title('Evolución del MAE en validación - LSTM')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Curvas_aprendizaje_LSTM.png', dpi=150)
    plt.savefig('Curvas_aprendizaje_LSTM.pdf')
    plt.close()

    return model, device, tiempos, best_val  # devolvemos también el mejor modelo y tiempos


def test_lstm(model, test_df, stats, device, seq_length=180):
    """Evalúa el modelo en test y devuelve el MAE en delta_log y log_mdr."""
    test_ds = TBDatasetLSTM(test_df, seq_length=seq_length)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    mae_delta, mae_mdr = evaluate_mae_real(model, test_loader, device, stats)
    return mae_delta, mae_mdr


if __name__ == "__main__":
    # Tiempo total del script (incluyendo carga y test)
    total_start = time.time()

    # 1. Preparación de datos (medir tiempo)
    prep_start = time.time()
    train_df, val_df, test_df, stats = preparar_datos(
        csv="dataset_tb.csv",
        stats_path="tb_target_stats.json",
        seed=42
    )
    prep_time = time.time() - prep_start

    # 2. Entrenamiento (ya mide sus tiempos internos)
    model, device, tiempos_entrenamiento, best_val_loss = train_lstm(
        train_df, val_df, stats,
        epochs=80, batch_size=64, lr=5e-4, weight_decay=1e-4,
        patience=20, w_delta=1.0, w_mdr=1.2, seq_length=180
    )

    # 3. Evaluación en test (usando el mejor modelo, que ya está cargado en 'model')
    test_start = time.time()
    test_mae_delta, test_mae_mdr = test_lstm(model, test_df, stats, device)
    test_time = time.time() - test_start

    total_time = time.time() - total_start

    # Mostrar resultados en test
    print("\nEvaluación en test")
    print(f"MAE delta_log (real): {test_mae_delta:.4f}")
    print(f"MAE log_mdr   (real): {test_mae_mdr:.4f}")

    # 4. Guardar tiempos en JSON
    times_dict = {
        "Preparación de datos": round(prep_time, 2),
        "Entrenamiento total": round(tiempos_entrenamiento["entrenamiento_total"], 2),
        "  - Épocas completadas": tiempos_entrenamiento["epocas_completadas"],
        "  - Tiempo promedio por época": round(tiempos_entrenamiento["tiempo_promedio_por_epoca"], 2),
        "Evaluación en test": round(test_time, 2),
        "TIEMPO TOTAL DE EJECUCIÓN": round(total_time, 2)
    }

    with open("times_lstm.json", "w") as f:
        json.dump(times_dict, f, indent=2)
