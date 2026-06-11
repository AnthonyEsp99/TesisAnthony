import numpy as np
import pandas as pd
from tb_model import TB_Model_Article_Normalized


def sample_treatment_4drugs(days=180, rng=None):
    
    if rng is None:
        rng = np.random.default_rng()

    p = rng.random()
    code = np.zeros(days, dtype=np.int64)

    # 1) Régimen intensivo con fases
    
    if p < 0.50:
        for d in range(days):
            if d < 60:
                x = rng.random()
                if x < 0.90:
                    code[d] = 15          # todos
                elif x < 0.95:
                    code[d] = 3           # INH + RIF
                elif x < 0.98:
                    code[d] = 12          # SRT + PZA
                else:
                    code[d] = rng.integers(0, 16)
            else:
                x = rng.random()
                if x < 0.60:
                    code[d] = 15
                elif x < 0.80:
                    code[d] = 3
                elif x < 0.90:
                    code[d] = 12
                else:
                    code[d] = rng.integers(0, 16)

    # 2) Mala adherencia con persistencia

    elif p < 0.80:
        prev = rng.integers(0, 16)
        for d in range(days):
            if rng.random() < 0.80:
                code[d] = prev
            else:
                prev = rng.integers(0, 16)
                code[d] = prev

    # 3) Esquemas fijos

    else:
        strategy = rng.integers(0, 4)

        if strategy == 0:
            code[:] = 15                     # siempre todos

        elif strategy == 1:
            mid = days // 2
            code[:mid] = 3                  # INH + RIF
            code[mid:] = 12                 # SRT + PZA

        elif strategy == 2:
            mid = days // 2
            code[:mid] = 12
            code[mid:] = 3

        else:
            for d in range(days):
                code[d] = 15 if (d % 2 == 0) else 0   # alternancia todos / nada

    return code


def generate_dataset_tb_master(n_samples=10000, days=180, seed=42, out_csv="dataset_tb.csv"):
    rng = np.random.default_rng(seed)
    rows = []
    failed = 0

    for i in range(n_samples):
        model = TB_Model_Article_Normalized()

        # Muestreo de parámetros alineado con tu tb_model actual
 
        model.beta_s = rng.uniform(0.36, 0.44)
        model.beta_r = rng.uniform(0.22, 0.30)

        model.delta_s = rng.uniform(0.30, 0.33)
        model.delta_r = rng.uniform(0.30, 0.34)

        model.q_H = 10 ** rng.uniform(-5.2, -4.8)   # ~ 1e-5
        model.q_R = 10 ** rng.uniform(-6.2, -5.8)   # ~ 1e-6
        model.q_S = 10 ** rng.uniform(-6.2, -5.8)   # ~ 1e-6
        model.q_Z = 10 ** rng.uniform(-4.2, -3.8)   # ~ 1e-4

        model.alpha_H = rng.uniform(0.030, 0.042)
        model.alpha_R = rng.uniform(0.030, 0.042)
        model.alpha_S = rng.uniform(0.022, 0.030)
        model.alpha_Z = rng.uniform(0.010, 0.014)
        # Las tasas mu se dejan como vienen de tb_model.py

     
        # Tratamiento y condiciones iniciales

        treat = sample_treatment_4drugs(days=days, rng=rng)

        s0 = rng.uniform(8e-3, 1.2e-2)
        r0 = rng.uniform(5e-8, 5e-7)

        # Debug opcional para revisar los primeros casos
        if i < 3:
            print(
                f"[DEBUG {i}] beta_s={model.beta_s:.4f}, beta_r={model.beta_r:.4f}, "
                f"alpha_H={model.alpha_H:.4f}, alpha_R={model.alpha_R:.4f}, "
                f"s0={s0:.6f}, r0={r0:.2e}"
            )

        try:
            _, y = model.simulate(treat.tolist(), days=days, s0=s0, r0=r0)
            s, r, _, _, _, _ = y
        except Exception:
            failed += 1
            continue

        total0 = s[0] + r[0]
        totalT = s[-1] + r[-1]

        log_total0 = np.log10(total0 + 1e-30)
        log_totalT = np.log10(totalT + 1e-30)
        delta_log = log_totalT - log_total0

        fraccion_mdr = r[-1] / (totalT + 1e-30)
        log_mdr = np.log10(fraccion_mdr + 1e-8)

        row = {
            "beta_s": model.beta_s,
            "beta_r": model.beta_r,
            "delta_s": model.delta_s,
            "delta_r": model.delta_r,
            "alpha_H": model.alpha_H,
            "alpha_R": model.alpha_R,
            "alpha_S": model.alpha_S,
            "alpha_Z": model.alpha_Z,
            "log_q_H": np.log10(model.q_H),
            "log_q_R": np.log10(model.q_R),
            "log_q_S": np.log10(model.q_S),
            "log_q_Z": np.log10(model.q_Z),
            "s0": s0,
            "r0": r0,
            "log_total0": log_total0,
            "log_totalT": log_totalT,
            "delta_log": delta_log,
            "fraccion_mdr": fraccion_mdr,
            "log_mdr": log_mdr,
        }

        for d in range(days):
            row[f"treat_{d}"] = int(treat[d])

        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{n_samples} | válidas: {len(rows)} | fallidas: {failed}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"\nDataset listo: {out_csv}")
    print(f"Válidas: {len(df)} | fallidas: {failed}")
    print(f"delta_log mean: {df['delta_log'].mean():.4f}")
    print(f"log_mdr mean: {df['log_mdr'].mean():.4f}")

    print("Proporción con delta_log < 0:", (df["delta_log"] < 0).mean())
    print("Proporción con delta_log < -0.2:", (df["delta_log"] < -0.2).mean())
    print("Proporción con delta_log < -1:", (df["delta_log"] < -1).mean())

    return df


if __name__ == "__main__":
    df = generate_dataset_tb_master(
        n_samples=10000,
        days=180,
        seed=42,
        out_csv="dataset_tb.csv"
    )
    print(df[["delta_log", "log_mdr"]].head())