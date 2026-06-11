import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_ode_model import TB_NeuralODE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class TBDataset(Dataset):
    STATIC_COLS = [
        "beta_s", "beta_r", "log_q_H", "log_q_R", "log_q_S", "log_q_Z",
        "s0", "r0", "delta_s", "delta_r", "alpha_H", "alpha_R", "alpha_S", "alpha_Z"
    ]

    def __init__(self, df: pd.DataFrame, seq_length: int = 180, scaler: Optional[StandardScaler] = None):
        self.df = df.reset_index(drop=True)
        self.seq_length = seq_length
        self.treat_cols = [f"treat_{d}" for d in range(seq_length)]

        # Validaciones
        missing_treat = [c for c in self.treat_cols if c not in self.df.columns]
        if missing_treat:
            raise ValueError(f"Faltan columnas treat_: {missing_treat[:5]}...")
        missing_static = [c for c in self.STATIC_COLS if c not in self.df.columns]
        if missing_static:
            raise ValueError(f"Faltan columnas estáticas: {missing_static}")
        for col in ["delta_log_z", "log_mdr_z"]:
            if col not in self.df.columns:
                raise ValueError(f"Falta columna objetivo normalizada: {col}")

        # Aplicar escalador si se proporciona
        static_values = self.df[self.STATIC_COLS].values.astype(np.float32)
        if scaler is not None:
            static_values = scaler.transform(static_values)
        self.static_data = static_values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        static = torch.tensor(self.static_data[idx], dtype=torch.float32)

        seq = torch.zeros(self.seq_length, 4, dtype=torch.float32)
        row = self.df.iloc[idx]
        for i, col in enumerate(self.treat_cols):
            code = int(row[col])
            seq[i, 0] = 1.0 if (code & 1) else 0.0
            seq[i, 1] = 1.0 if (code & 2) else 0.0
            seq[i, 2] = 1.0 if (code & 4) else 0.0
            seq[i, 3] = 1.0 if (code & 8) else 0.0

        y_delta_z = torch.tensor(row["delta_log_z"], dtype=torch.float32)
        y_mdr_z = torch.tensor(row["log_mdr_z"], dtype=torch.float32)
        return static, seq, y_delta_z, y_mdr_z


def normalize_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    mu_delta = float(df["delta_log"].mean())
    sd_delta = float(df["delta_log"].std() + 1e-8)
    mu_mdr = float(df["log_mdr"].mean())
    sd_mdr = float(df["log_mdr"].std() + 1e-8)

    df = df.copy()
    df["delta_log_z"] = (df["delta_log"] - mu_delta) / sd_delta
    df["log_mdr_z"] = (df["log_mdr"] - mu_mdr) / sd_mdr

    stats = {"mu_delta": mu_delta, "sd_delta": sd_delta, "mu_mdr": mu_mdr, "sd_mdr": sd_mdr}
    return df, stats


def preparar_datos(
    csv_path: str,
    test_size: float = 0.3,
    val_size: float = 0.5,
    seed: int = 42,
    save_stats_path: str = "tb_target_stats.json",
    save_scaler_path: str = "static_scaler.pkl"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float], StandardScaler]:
    logger.info(f"Cargando dataset desde {csv_path}...")
    t_start = time.perf_counter()
    df = pd.read_csv(csv_path)
    # df = df.head(100)  # Opcional para pruebas, quitar cuando se use el dataset completo

    required_cols = [
        "delta_log", "log_mdr",
        "beta_s", "beta_r", "log_q_H", "log_q_R", "log_q_S", "log_q_Z",
        "s0", "r0", "delta_s", "delta_r", "alpha_H", "alpha_R", "alpha_S", "alpha_Z"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el CSV: {missing}")

    treat_cols = [c for c in df.columns if c.startswith("treat_")]
    logger.info(f"Dataset: {len(df)} filas | Columnas treat: {len(treat_cols)}")

    # Normalizar objetivos
    df, stats = normalize_targets(df)

    # Dividir
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=seed)

    # Normalizar características estáticas (solo con entrenamiento)
    static_cols = TBDataset.STATIC_COLS
    scaler = StandardScaler()
    scaler.fit(train_df[static_cols].values)
    import joblib
    joblib.dump(scaler, save_scaler_path)
    logger.info(f"Scaler de estáticas guardado en {save_scaler_path}")

    # Verificar normalización de objetivos
    logger.info(f"Train: delta_log_z media={train_df['delta_log_z'].mean():.4f}, std={train_df['delta_log_z'].std():.4f}")
    logger.info(f"Train: log_mdr_z media={train_df['log_mdr_z'].mean():.4f}, std={train_df['log_mdr_z'].std():.4f}")

    # Guardar estadísticas de objetivos
    Path(save_stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Estadísticas de normalización guardadas en {save_stats_path}")

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    t_end = time.perf_counter()
    logger.info(f"Tiempo de preparación de datos: {t_end - t_start:.2f} segundos")

    return train_df, val_df, test_df, stats, scaler


def evaluate_mae_real(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stats: Dict[str, float],
    dt: float,
    method: str
) -> Tuple[float, float]:
    model.eval()
    mae_delta_z = 0.0
    mae_mdr_z = 0.0
    n = 0
    with torch.no_grad():
        for static, seq, y_delta_z, y_mdr_z in loader:
            static = static.to(device, non_blocking=True)
            seq = seq.to(device, non_blocking=True)
            y_delta_z = y_delta_z.to(device, non_blocking=True)
            y_mdr_z = y_mdr_z.to(device, non_blocking=True)

            p_delta_z, p_mdr_z = model(static, seq, dt=dt, method=method)

            mae_delta_z += torch.abs(p_delta_z - y_delta_z).sum().item()
            mae_mdr_z += torch.abs(p_mdr_z - y_mdr_z).sum().item()
            n += y_delta_z.size(0)

    mae_delta_real = (mae_delta_z / n) * stats["sd_delta"]
    mae_mdr_real = (mae_mdr_z / n) * stats["sd_mdr"]
    return mae_delta_real, mae_mdr_real


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_neural_ode(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stats: Dict[str, float],
    scaler: StandardScaler,
    latent_dim: int = 256,
    method: str = 'dopri5',
    dt: float = 0.05,
    seq_length: int = 180,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    w_delta: float = 1.0,
    w_mdr: float = 4.0,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
    use_amp: bool = False,
    save_best_path: str = "tb_neural_ode_best.pth",
    save_last_path: str = "tb_neural_ode_last.pth",
) -> Tuple[nn.Module, torch.device, Dict[str, float]]:
    tiempos = {}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de entrenamiento: {device}")

    logger.info("Creando datasets y dataloaders...")
    t_data = time.perf_counter()
    train_ds = TBDataset(train_df, seq_length=seq_length, scaler=scaler)
    val_ds = TBDataset(val_df, seq_length=seq_length, scaler=scaler)
    test_ds = TBDataset(test_df, seq_length=seq_length, scaler=scaler)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    tiempos["data_preparation"] = time.perf_counter() - t_data

    model = TB_NeuralODE(latent_dim=latent_dim, static_dim=14, control_dim=4).to(device)
    model.apply(init_weights)  # Inicialización de pesos

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4  # , verbose=True
    )

    mse_loss = nn.MSELoss()
    huber_loss = nn.SmoothL1Loss(beta=1.0)

    scaler_amp = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    best_val_loss = float('inf')
    wait = 0
    best_model_state = None

    print("\nConfiguración de entrenamiento Neural ODE")
    print(f"epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"weight_decay={weight_decay}, dt={dt}, patience={patience}")
    print(f"w_delta={w_delta}, w_mdr={w_mdr}, latent_dim={latent_dim}")
    print(f"method={method}, use_amp={use_amp and scaler_amp is not None}")

    t_train_start = time.perf_counter()
    epoch_times = []
    train_losses = []
    val_losses = []
    val_mae_delta = []
    val_mae_mdr = []

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()

        model.train()
        train_loss = 0.0
        for static, seq, y_delta_z, y_mdr_z in train_loader:
            static = static.to(device, non_blocking=True)
            seq = seq.to(device, non_blocking=True)
            y_delta_z = y_delta_z.to(device, non_blocking=True)
            y_mdr_z = y_mdr_z.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler_amp is not None:
                with torch.cuda.amp.autocast():
                    p_delta_z, p_mdr_z = model(static, seq, dt=dt, method=method)
                    loss_delta = mse_loss(p_delta_z, y_delta_z)
                    loss_mdr = huber_loss(p_mdr_z, y_mdr_z)
                    loss = w_delta * loss_delta + w_mdr * loss_mdr
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                p_delta_z, p_mdr_z = model(static, seq, dt=dt, method=method)
                loss_delta = mse_loss(p_delta_z, y_delta_z)
                loss_mdr = huber_loss(p_mdr_z, y_mdr_z)
                loss = w_delta * loss_delta + w_mdr * loss_mdr
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for static, seq, y_delta_z, y_mdr_z in val_loader:
                static = static.to(device, non_blocking=True)
                seq = seq.to(device, non_blocking=True)
                y_delta_z = y_delta_z.to(device, non_blocking=True)
                y_mdr_z = y_mdr_z.to(device, non_blocking=True)

                p_delta_z, p_mdr_z = model(static, seq, dt=dt, method=method)
                loss_delta = mse_loss(p_delta_z, y_delta_z)
                loss_mdr = huber_loss(p_mdr_z, y_mdr_z)
                loss = w_delta * loss_delta + w_mdr * loss_mdr
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        mae_delta_val, mae_mdr_val = evaluate_mae_real(model, val_loader, device, stats, dt, method)
        val_mae_delta.append(mae_delta_val)
        val_mae_mdr.append(mae_mdr_val)

        scheduler.step(val_loss)

        t_epoch_end = time.perf_counter()
        epoch_time = t_epoch_end - t_epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch}/{epochs} ({epoch_time:.2f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val MAE delta_log (real): {mae_delta_val:.4f}")
        print(f"  Val MAE log_mdr   (real): {mae_mdr_val:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            wait = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), save_best_path)
            print(f"  -> Nuevo mejor modelo guardado (loss: {best_val_loss:.4f})")
        else:
            wait += 1
            print(f"  -> Sin mejora ({wait}/{patience})")
            if wait >= patience:
                print("Early stopping activado.")
                break

    t_train_end = time.perf_counter()
    tiempos["training_total"] = t_train_end - t_train_start
    tiempos["epochs_completed"] = len(epoch_times)
    tiempos["avg_epoch_time"] = np.mean(epoch_times) if epoch_times else 0

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Mejor modelo restaurado.")

    # Gráficas de aprendizaje - guardar como PNG y PDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label='Entrenamiento', linewidth=2)
    ax1.plot(val_losses, label='Validación', linewidth=2)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida (MSE + Huber)')
    ax1.set_title('Curvas de pérdida')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_mae_delta, label='MAE Δlog (real)', linewidth=2)
    ax2.plot(val_mae_mdr, label='MAE log MDR (real)', linewidth=2)
    ax2.set_xlabel('Época')
    ax2.set_ylabel('MAE (escala real)')
    ax2.set_title('Evolución del MAE en validación')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Curvas_aprendizaje.png', dpi=150)
    plt.savefig('Curvas_aprendizaje.pdf')
    plt.close()
    print("Curvas de aprendizaje guardadas en Curvas_aprendizaje.png y Curvas_aprendizaje.pdf")

    torch.save(model.state_dict(), save_last_path)
    print(f"Modelo final guardado en {save_last_path}")
    print(f"Mejor val loss: {best_val_loss:.4f}")

    print("\n=== Evaluación en conjunto de test ===")
    t_test_start = time.perf_counter()
    test_mae_delta, test_mae_mdr = evaluate_mae_real(model, test_loader, device, stats, dt, method)
    t_test_end = time.perf_counter()
    tiempos["test_evaluation"] = t_test_end - t_test_start

    print(f"Test MAE delta_log (real): {test_mae_delta:.4f}")
    print(f"Test MAE log_mdr   (real): {test_mae_mdr:.4f}")

    # Guardar métricas resumidas
    metrics_summary = {
        "best_val_loss": best_val_loss,
        "test_mae_delta": test_mae_delta,
        "test_mae_mdr": test_mae_mdr,
        "epochs_completed": len(epoch_times),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_mae_delta": val_mae_delta,
        "val_mae_mdr": val_mae_mdr
    }
    with open("metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print("Métricas guardadas en metrics_summary.json")

    return model, device, tiempos


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de Neural ODE para TB con métricas mejoradas")
    parser.add_argument("--csv", type=str, default="dataset_tb.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seq_length", type=int, default=180)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--w_delta", type=float, default=1.0)
    parser.add_argument("--w_mdr", type=float, default=4.0)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--method", type=str, default="dopri5", choices=["euler", "rk4", "dopri5"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--save_stats", type=str, default="tb_target_stats.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t_total_start = time.perf_counter()

    train_df, val_df, test_df, stats, scaler = preparar_datos(
        args.csv, test_size=0.3, val_size=0.5, seed=args.seed,
        save_stats_path=args.save_stats, save_scaler_path="static_scaler.pkl"
    )

    model, device, tiempos = train_neural_ode(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        stats=stats,
        scaler=scaler,
        latent_dim=args.latent_dim,
        method=args.method,
        dt=args.dt,
        seq_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        w_delta=args.w_delta,
        w_mdr=args.w_mdr,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
    )

    t_total_end = time.perf_counter()
    tiempos["total_execution"] = t_total_end - t_total_start

    print("\n" + "=" * 60)
    print("RESUMEN DE TIEMPOS DE EJECUCIÓN")
    print("=" * 60)
    print(f"Preparación de datos:          {tiempos.get('data_preparation', 0):.2f} s")
    print(f"Entrenamiento total:            {tiempos['training_total']:.2f} s")
    print(f"  - Épocas completadas:         {tiempos['epochs_completed']}")
    print(f"  - Tiempo promedio por época:  {tiempos['avg_epoch_time']:.2f} s")
    print(f"Evaluación en test:             {tiempos['test_evaluation']:.2f} s")
    print(f"TIEMPO TOTAL DE EJECUCIÓN:      {tiempos['total_execution']:.2f} s")
    print("=" * 60)

    with open("execution_times.json", "w") as f:
        json.dump(tiempos, f, indent=2)
    print("Tiempos guardados en execution_times.json")


if __name__ == "__main__":
    main()