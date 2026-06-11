import os
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt
from neural_ode_model import TB_NeuralODE
from tb_model import TB_Model_Article_Normalized

# 1. Modelo simple de clasificación (cura)

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
        return x

# 2. Cargar modelos y estadísticas

def load_simple_model(model_path, input_dim, device):
    model = SimpleClassifier(input_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Modelo de cura cargado: {model_path} (input_dim={input_dim})")
    return model

def predict_proba_simple(model, X, device, batch_size=256):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch)
            probs_batch = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.append(probs_batch)
    return np.concatenate(probs)

def load_ode_model(model_path, latent_dim=256, static_dim=14, control_dim=4, device='cpu'):
    model = TB_NeuralODE(latent_dim=latent_dim, static_dim=static_dim, control_dim=control_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Modelo ODE cargado: {model_path}")
    return model

def load_stats(stats_path):
    with open(stats_path, "r") as f:
        stats = json.load(f)
    if 'mu_mdr' not in stats:
        stats['mu_mdr'] = 0.0
        stats['sd_mdr'] = 1.0
        print("Advertencia: stats no contiene mu_mdr/sd_mdr, usando valores por defecto (0, 1).")
    print(f"Estadísticas cargadas: mu_delta={stats['mu_delta']:.3f}, sd_delta={stats['sd_delta']:.3f}, "  #Para probar si carga
          f"mu_mdr={stats['mu_mdr']:.3f}, sd_mdr={stats['sd_mdr']:.3f}")
    return stats

# 3. Utilidades para secuencias

def secuencias_a_onehot(secuencias):
    B, T = secuencias.shape
    onehot = np.zeros((B, T, 4), dtype=np.float32)
    for i in range(B):
        for t, code in enumerate(secuencias[i]):
            code = int(code) & 0xF
            onehot[i, t, 0] = 1.0 if (code & 1) else 0.0
            onehot[i, t, 1] = 1.0 if (code & 2) else 0.0
            onehot[i, t, 2] = 1.0 if (code & 4) else 0.0
            onehot[i, t, 3] = 1.0 if (code & 8) else 0.0
    return torch.tensor(onehot, dtype=torch.float32)

def codigos_a_farmacos(codigos):
    farmacos_por_dia = []
    for code in codigos:
        combo = []
        if code & 1: combo.append("INH")
        if code & 2: combo.append("RIF")
        if code & 4: combo.append("SRT")
        if code & 8: combo.append("PZA")
        farmacos_por_dia.append(combo if combo else "Ninguno")
    return farmacos_por_dia

def resumir_farmacos(farmacos_por_dia):
    flat = [f for dia in farmacos_por_dia if isinstance(dia, list) for f in dia]
    return Counter(flat)

def guardar_tratamiento_completo(seq_codigos, paciente_id, output_dir):

    #Guarda la secuencia de tratamiento en un único archivo CSV:
    #columnas: dia, codigo, farmacos

    subdir = os.path.join(output_dir, "tratamientos_optimizados")
    os.makedirs(subdir, exist_ok=True)
    base = f"paciente_{paciente_id}_tratamiento.csv"
    filepath = os.path.join(subdir, base)
    farmacos = codigos_a_farmacos(seq_codigos)
    farmacos_str = [", ".join(f) if isinstance(f, list) else f for f in farmacos]
    df = pd.DataFrame({
        "dia": range(1, len(seq_codigos)+1),
        "codigo": seq_codigos,
        "farmacos": farmacos_str
    })
    df.to_csv(filepath, index=False)
    print(f"Tratamiento completo para paciente guardado {paciente_id}: {filepath}")
    return filepath


# 4. Algoritmo genético

def evaluar_poblacion(ode_model, stats, static_tensor, poblacion, device,
                      w1=1.0, w2=2.0, penalizacion_resistencia=None):
    B, T = poblacion.shape
    seq_tensor = secuencias_a_onehot(poblacion).to(device)
    static_batch = static_tensor.repeat(B, 1)
    with torch.no_grad():
        p_delta_z, p_mdr_z = ode_model(static_batch, seq_tensor)
        delta = p_delta_z.cpu().numpy() * stats['sd_delta'] + stats['mu_delta']
        log_mdr = p_mdr_z.cpu().numpy() * stats['sd_mdr'] + stats['mu_mdr']
    
    if penalizacion_resistencia is not None:
        umbral, lam = penalizacion_resistencia
        penal = lam * np.maximum(0, log_mdr - umbral)
    else:
        penal = 0.0
    
    costo = w1 * delta + w2 * log_mdr + penal
    return costo, delta, log_mdr

def cruzar(padre, madre):
    punto = np.random.randint(1, len(padre)-1)
    hijo1 = np.concatenate([padre[:punto], madre[punto:]])
    hijo2 = np.concatenate([madre[:punto], padre[punto:]])
    return hijo1, hijo2

def mutar(seq, prob_mutacion=0.04):
    for i in range(len(seq)):
        if np.random.random() < prob_mutacion:
            seq[i] = np.random.randint(0, 16)
    return seq

def recomendar_mejor_tratamiento_genetico(ode_model, stats, static_tensor,
                                          poblacion_size=400, generaciones=60,
                                          prob_mutacion=0.04, device='cpu',
                                          w1=1.0, w2=2.0, elite_frac=0.05,
                                          penalizacion_resistencia=None):
    T = 180
    # Semillas clínicas
    semillas = []
    semillas.append(np.full(T, 15, dtype=np.int64))
    s2 = np.full(T, 3, dtype=np.int64)
    s2[:60] = 15
    semillas.append(s2)
    s3 = np.full(T, 12, dtype=np.int64)
    s3[:60] = 15
    semillas.append(s3)
    s4 = np.zeros(T, dtype=np.int64)
    s4[0::2] = 15
    semillas.append(s4)
    
    poblacion = list(semillas)
    restantes = poblacion_size - len(semillas)
    if restantes > 0:
        poblacion.extend(np.random.randint(0, 16, size=(restantes, T)))
    poblacion = np.array(poblacion[:poblacion_size])
    
    costo, delta, log_mdr = evaluar_poblacion(ode_model, stats, static_tensor, poblacion, device,
                                               w1, w2, penalizacion_resistencia)
    mejor_idx = np.argmin(costo)
    mejor_costo = costo[mejor_idx]
    mejor_delta = delta[mejor_idx]
    mejor_log_mdr = log_mdr[mejor_idx]
    mejor_seq = poblacion[mejor_idx].copy()
    
    candidatos = [(costo[i], delta[i], log_mdr[i], poblacion[i].copy()) for i in range(poblacion_size)]
    num_elite = max(1, int(elite_frac * poblacion_size))
    
    for gen in range(generaciones):
        nueva_poblacion = []
        idx_elite = np.argsort(costo)[:num_elite]
        for i in idx_elite:
            nueva_poblacion.append(poblacion[i].copy())
        
        while len(nueva_poblacion) < poblacion_size:
            idx1 = np.random.choice(poblacion_size, 3, replace=False)
            idx2 = np.random.choice(poblacion_size, 3, replace=False)
            padre = poblacion[np.argmin(costo[idx1])]
            madre = poblacion[np.argmin(costo[idx2])]
            hijo1, hijo2 = cruzar(padre, madre)
            hijo1 = mutar(hijo1, prob_mutacion)
            hijo2 = mutar(hijo2, prob_mutacion)
            nueva_poblacion.extend([hijo1, hijo2])
        poblacion = np.array(nueva_poblacion[:poblacion_size])
        costo, delta, log_mdr = evaluar_poblacion(ode_model, stats, static_tensor, poblacion, device,
                                                   w1, w2, penalizacion_resistencia)
        mejor_actual = np.argmin(costo)
        if costo[mejor_actual] < mejor_costo:
            mejor_costo = costo[mejor_actual]
            mejor_delta = delta[mejor_actual]
            mejor_log_mdr = log_mdr[mejor_actual]
            mejor_seq = poblacion[mejor_actual].copy()
        candidatos.extend([(costo[i], delta[i], log_mdr[i], poblacion[i].copy()) for i in range(poblacion_size)])
    
    candidatos.sort(key=lambda x: x[0])
    top10 = candidatos[:10]
    return mejor_seq, mejor_delta, mejor_log_mdr, top10

# 5. Simulación con modelo explícito (gráficas)

def simular_ode_explicito(static_dict, seq_codigos, days=180):
    ode = TB_Model_Article_Normalized()
    ode.beta_s = static_dict["beta_s"]
    ode.beta_r = static_dict["beta_r"]
    ode.delta_s = static_dict["delta_s"]
    ode.delta_r = static_dict["delta_r"]
    ode.q_H = 10 ** static_dict["log_q_H"]
    ode.q_R = 10 ** static_dict["log_q_R"]
    ode.q_S = 10 ** static_dict["log_q_S"]
    ode.q_Z = 10 ** static_dict["log_q_Z"]
    ode.alpha_H = static_dict["alpha_H"]
    ode.alpha_R = static_dict["alpha_R"]
    ode.alpha_S = static_dict["alpha_S"]
    ode.alpha_Z = static_dict["alpha_Z"]
    s0 = static_dict["s0"]
    r0 = static_dict["r0"]
    t, y = ode.simulate(seq_codigos, days=days, s0=s0, r0=r0)
    s, r = y[0], y[1]
    total = s + r
    frac_res = r / (total + 1e-12)
    return t, total, frac_res


# 6. Preparación de características

STATIC_COLS_ODE = [
    "beta_s", "beta_r", "log_q_H", "log_q_R", "log_q_S", "log_q_Z",
    "s0", "r0", "delta_s", "delta_r", "alpha_H", "alpha_R", "alpha_S", "alpha_Z"
]

def obtener_features_simple(df, input_dim):
    exclude = ["delta_log", "log_mdr", "cure", "id", "patient_id"]
    numeric_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    if len(numeric_cols) < input_dim:
        raise ValueError(f"Se requieren {input_dim} columnas numéricas, solo hay {len(numeric_cols)}")
    feature_cols = numeric_cols[:input_dim]
    X = df[feature_cols].values.astype(np.float32)
    return X, feature_cols

def obtener_features_ode(df):
    missing = [c for c in STATIC_COLS_ODE if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para modelo ODE: {missing}")
    static = df[STATIC_COLS_ODE].values.astype(np.float32)
    return static


# 7. Gráficas (para un paciente dado)

def simular_trayectoria_ode(ode_model, stats, static_tensor, seq_codigos, device='cpu', T=180):
    from torchdiffeq import odeint
    seq_tensor = secuencias_a_onehot(np.array([seq_codigos])).to(device)
    x0 = ode_model.encoder(static_tensor)
    def u_func(t):
        day = torch.floor(t).long().clamp(0, T-1)
        return seq_tensor[:, day, :]
    def func(t, x):
        return ode_model.odefunc(t, x, static_tensor, u_func)
    t_span = torch.linspace(0.0, float(T), steps=100, device=device)
    x_traj = odeint(func, x0, t_span, method='dopri5', rtol=1e-5, atol=1e-6)
    delta_logs, log_mdrs = [], []
    for x in x_traj:
        out = ode_model.decoder(torch.cat([x, static_tensor], dim=1))
        delta_z, mdr_z = out[:, 0], out[:, 1]
        delta = delta_z.item() * stats['sd_delta'] + stats['mu_delta']
        mdr = mdr_z.item() * stats['sd_mdr'] + stats['mu_mdr']
        delta_logs.append(delta)
        log_mdrs.append(mdr)
    return t_span.cpu().numpy(), np.array(delta_logs), np.array(log_mdrs)

def graficar_comparativa_tratamientos(tiempo, lista_deltas, lista_mdrs, nombres_tratamientos, output_path_base, tau):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for i, (delta, nombre) in enumerate(zip(lista_deltas, nombres_tratamientos)):
        plt.plot(tiempo, delta, linewidth=1.5, label=f"{nombre}")
    plt.axhline(y=tau, color='r', linestyle='--', label=f'tau = {tau}')
    plt.xlabel('Días'); plt.ylabel('delta_log')
    plt.title('Evolución de la carga bacteriana (ODE)')
    plt.legend(loc='best', fontsize=8); plt.grid(True)
    plt.subplot(1, 2, 2)
    for i, (mdr, nombre) in enumerate(zip(lista_mdrs, nombres_tratamientos)):
        plt.plot(tiempo, mdr, linewidth=1.5, label=f"{nombre}")
    plt.xlabel('Días'); plt.ylabel('log_mdr')
    plt.title('Evolución de la resistencia (ODE)')
    plt.legend(loc='best', fontsize=8); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_path_base}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_path_base}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Gráficas ODE guardadas en {output_path_base}.png y .pdf")

def generar_graficas_paciente(static_tensor, static_dict, top_candidatos, output_dir, ode_model, stats, device, tau, paciente_id=None, sufijo=""):
    if not top_candidatos:
        print(f"  No hay candidatos para paciente {paciente_id}")
        return
    
    print(f"\n=== 10 MEJORES TRATAMIENTOS (paciente {paciente_id}) ===")
    for idx, (costo_val, delta_val, log_mdr_val, seq) in enumerate(top_candidatos[:10]):
        farmacos_por_dia = codigos_a_farmacos(seq)
        freq = resumir_farmacos(farmacos_por_dia)
        resumen = ", ".join([f"{k}: {v}d" for k, v in freq.items()])
        print(f"{idx+1}. costo={costo_val:.4f} delta={delta_val:.4f} log_mdr={log_mdr_val:.4f} -> {resumen}")
    
    # Curvas ODE
    tiempos = None
    lista_deltas, lista_mdrs, nombres = [], [], []
    for idx, (costo_val, delta_val, log_mdr_val, seq) in enumerate(top_candidatos[:10]):
        farmacos_por_dia = codigos_a_farmacos(seq)
        freq = resumir_farmacos(farmacos_por_dia)
        resumen = ", ".join([f"{k}: {v}d" for k, v in freq.items()])[:40]
        nombres.append(f"#{idx+1}: {resumen}")
        t, delta_traj, mdr_traj = simular_trayectoria_ode(ode_model, stats, static_tensor, seq, device)
        if tiempos is None: tiempos = t
        lista_deltas.append(delta_traj)
        lista_mdrs.append(mdr_traj)
    base_ode = os.path.join(output_dir, f"evolucion_tratamientos{sufijo}")
    graficar_comparativa_tratamientos(tiempos, lista_deltas, lista_mdrs, nombres, base_ode, tau)
    
    # Gráficas explícitas
    print(f"\nGenerando gráficas con modelo explícito para paciente {paciente_id}...")
    all_total, all_resist = [], []
    for idx, (costo_val, delta_val, log_mdr_val, seq) in enumerate(top_candidatos[:10]):
        try:
            t, total, frac_res = simular_ode_explicito(static_dict, seq)
            label = f"Tratamiento {idx+1} (delta={delta_val:.3f}, logMDR={log_mdr_val:.3f})"
            all_total.append((t, total, label))
            all_resist.append((t, frac_res, label))
        except Exception as e:
            print(f"Error en tratamiento {idx+1}: {e}")
    if all_total:
        plt.figure(figsize=(12,8))
        for t, total, label in all_total:
            plt.plot(t, total, linewidth=1.5, label=label)
        plt.xlabel("Días"); plt.ylabel("Carga bacteriana total (S+R)")
        plt.title(f"Evolución de la carga bacteriana (modelo explícito) - Paciente {paciente_id}")
        plt.legend(fontsize=8); plt.grid(True)
        total_path = os.path.join(output_dir, f"carga_bacteriana_combinada{sufijo}")
        plt.savefig(f"{total_path}.png", dpi=150); plt.savefig(f"{total_path}.pdf"); plt.close()
        print(f"Guardada gráfica de carga bacteriana en {total_path}.pdf")
    if all_resist:
        plt.figure(figsize=(12,8))
        for t, frac_res, label in all_resist:
            plt.plot(t, frac_res, linewidth=1.5, label=label)
        plt.xlabel("Días"); plt.ylabel("Fracción de resistentes")
        plt.title(f"Evolución de la resistencia (modelo explícito) - Paciente {paciente_id}")
        plt.legend(fontsize=8); plt.grid(True)
        resist_path = os.path.join(output_dir, f"resistencia_combinada{sufijo}")
        plt.savefig(f"{resist_path}.png", dpi=150); plt.savefig(f"{resist_path}.pdf"); plt.close()
        print(f"Guardada gráfica de resistencia en {resist_path}.pdf")


# 8. Función principal (modificada)

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Recomendación de tratamiento con algoritmo genético multiobjetivo")
    parser.add_argument("--input", type=str, default=None, help="Archivo CSV con datos de pacientes")
    parser.add_argument("--model_cure", type=str, default="tb_neural_ode_cure_best.pth", help="Modelo clasificador simple")
    parser.add_argument("--model_ode", type=str, default="tb_neural_ode_best.pth", help="Modelo ODE")
    parser.add_argument("--stats", type=str, default="tb_target_stats.json", help="Estadísticas de normalización")
    parser.add_argument("--output_dir", type=str, default="MisDatos", help="Directorio de salida")
    parser.add_argument("--threshold", type=float, default=0.317, help="Umbral clasificador simple para recomendar tratamiento")
    parser.add_argument("--tau", type=float, default=-1.7, help="Umbral de cura (delta_log <= tau se considera cura)")
    parser.add_argument("--poblacion", type=int, default=400, help="Tamaño de la población")
    parser.add_argument("--generaciones", type=int, default=60, help="Número de generaciones")
    parser.add_argument("--mutacion", type=float, default=0.04, help="Probabilidad de mutación por gen")
    parser.add_argument("--elite_frac", type=float, default=0.05, help="Fracción de élite que se conserva")
    parser.add_argument("--w1", type=float, default=2.0, help="Peso para delta_log en la función costo")
    parser.add_argument("--w2", type=float, default=2.0, help="Peso para log_mdr en la función costo")
    parser.add_argument("--penal_umbral", type=float, default=None, help="Umbral de log_mdr para penalización (ej. -2.0)")
    parser.add_argument("--penal_lambda", type=float, default=10.0, help="Factor de penalización si se supera umbral")
    parser.add_argument("--input_dim", type=int, default=20, help="Dimensión de entrada del clasificador simple")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, "resultados_tratamiento.csv")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Probabilidad de mutación: {args.mutacion}")

    # Asegurar archivo de entrada (por defecto usar solo 2 pacientes)
    if args.input is None or not os.path.exists(args.input):
        print("No se especificó --input o no existe. Se crean 2 muestras desde dataset_tb.csv.")
        if not os.path.exists("dataset_tb.csv"):
            raise FileNotFoundError("dataset_tb.csv no encontrado")
        df_original = pd.read_csv("dataset_tb.csv")
        # CAMBIA AQUÍ EL NÚMERO DE PACIENTES POR DEFECTO (2)
        df_sample = df_original.head(200).copy()
        exclude_cols = ["delta_log", "log_mdr"]
        numeric_cols = [c for c in df_sample.columns if c not in exclude_cols and np.issubdtype(df_sample[c].dtype, np.number)]
        df_sample = df_sample[numeric_cols]
        input_csv = os.path.join(args.output_dir, "nuevos_pacientes.csv")
        df_sample.to_csv(input_csv, index=False)
        print(f"Archivo de ejemplo creado con 2 pacientes: {input_csv}")
    else:
        input_csv = args.input

    df = pd.read_csv(input_csv)
    print(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")

    # Clasificador simple
    if args.input_dim is None:
        state = torch.load(args.model_cure, map_location="cpu")
        input_dim = state["fc1.weight"].shape[1]
    else:
        input_dim = args.input_dim
    X_simple, _ = obtener_features_simple(df, input_dim)
    model_cure = load_simple_model(args.model_cure, input_dim, device)
    prob_cura = predict_proba_simple(model_cure, X_simple, device)
    recomendacion_cura = np.where(prob_cura < args.threshold, "Tratamiento recomendado", "Seguimiento sin tratamiento")

    # Modelo ODE
    ode_model = load_ode_model(args.model_ode, device=device)
    stats = load_stats(args.stats)
    static_ode = obtener_features_ode(df)
    print(f"Características ODE: {static_ode.shape}")

    penalizacion = None
    if args.penal_umbral is not None:
        penalizacion = (args.penal_umbral, args.penal_lambda)
        print(f"Usando penalización de resistencia: umbral={args.penal_umbral}, lambda={args.penal_lambda}")

    recomendaciones = []
    delta_logs = []
    log_mdrs = []
    curacion_segun_ode = []
    archivos_tratamiento = []

    # Variables para detectar pacientes curados y no curados
    paciente_cura_idx = None
    paciente_cura_static_tensor = None
    paciente_cura_static_dict = None
    paciente_cura_top_candidatos = None
    paciente_no_curado_idx = None
    paciente_no_curado_static_tensor = None
    paciente_no_curado_static_dict = None
    paciente_no_curado_top_candidatos = None

    # Primer paciente (para mantener compatibilidad)
    top_candidatos_primer_paciente = None
    static_primer_paciente = None
    static_dict_primer_paciente = None

    print("\n=== Buscando mejor tratamiento con algoritmo genético multiobjetivo ===")
    num_pacientes = len(df)
    for i in range(num_pacientes):
        print(f"Paciente {i+1}/{num_pacientes}")
        static_tensor = torch.tensor(static_ode[i:i+1], dtype=torch.float32).to(device)
        mejor_seq, mejor_delta, mejor_log_mdr, top_candidatos = recomendar_mejor_tratamiento_genetico(
            ode_model, stats, static_tensor,
            poblacion_size=args.poblacion,
            generaciones=args.generaciones,
            prob_mutacion=args.mutacion,
            device=device,
            w1=args.w1,
            w2=args.w2,
            elite_frac=args.elite_frac,
            penalizacion_resistencia=penalizacion
        )
        ruta_tratamiento = guardar_tratamiento_completo(mejor_seq, paciente_id=i, output_dir=args.output_dir)
        archivos_tratamiento.append(ruta_tratamiento)

        if i == 0:
            top_candidatos_primer_paciente = top_candidatos
            static_primer_paciente = static_tensor
            first_row = df.iloc[0]
            static_dict_primer_paciente = {col: first_row[col] for col in STATIC_COLS_ODE}

        # Detectar primer paciente CURADO
        if paciente_cura_idx is None and (mejor_delta <= args.tau):
            paciente_cura_idx = i
            paciente_cura_static_tensor = static_tensor
            paciente_cura_static_dict = {col: df.iloc[i][col] for col in STATIC_COLS_ODE}
            paciente_cura_top_candidatos = top_candidatos
            print(f"  >>> Paciente CURADO detectado: índice {i}, delta={mejor_delta:.4f} <= tau={args.tau}")

        # Detectar primer paciente NO CURADO
        if paciente_no_curado_idx is None and (mejor_delta > args.tau):
            paciente_no_curado_idx = i
            paciente_no_curado_static_tensor = static_tensor
            paciente_no_curado_static_dict = {col: df.iloc[i][col] for col in STATIC_COLS_ODE}
            paciente_no_curado_top_candidatos = top_candidatos
            print(f"  !!! Paciente NO CURADO detectado: índice {i}, delta={mejor_delta:.4f} > tau={args.tau}")

        farmacos_por_dia = codigos_a_farmacos(mejor_seq)
        freq = resumir_farmacos(farmacos_por_dia)
        resumen = ", ".join([f"{k}: {v} días" for k, v in freq.items()])
        recomendaciones.append(resumen if resumen else "Ninguno")
        delta_logs.append(mejor_delta)
        log_mdrs.append(mejor_log_mdr)
        es_cura = (mejor_delta <= args.tau)
        curacion_segun_ode.append(es_cura)

    # Calcular métricas globales
    promedio_delta = np.mean(delta_logs)
    promedio_log_mdr = np.mean(log_mdrs)
    porcentaje_cura = 100.0 * np.mean(curacion_segun_ode)
    tiempo_total = time.time() - start_time

    # Guardar parámetros en JSON
    params = {
        "tau": args.tau,
        "promedio_delta_log": float(promedio_delta),
        "promedio_log_mdr": float(promedio_log_mdr),
        "porcentaje_cura": float(porcentaje_cura),
        "tiempo_total_segundos": tiempo_total,
        "num_pacientes": len(df),
        "algoritmo_genetico": {
            "poblacion": args.poblacion,
            "generaciones": args.generaciones,
            "mutacion": args.mutacion,
            "elite_frac": args.elite_frac
        },
        "pesos_costo": {"w1": args.w1, "w2": args.w2},
        "penalizacion_resistencia": penalizacion
    }
    params_path = os.path.join(args.output_dir, "parametros_tratamientos.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nParámetros globales guardados en {params_path}")

    # Guardar resultados CSV
    df_resultado = df.copy()
    df_resultado["prob_cura"] = prob_cura
    df_resultado["recomendacion_cura"] = recomendacion_cura
    df_resultado["mejor_delta_log"] = delta_logs
    df_resultado["mejor_log_mdr"] = log_mdrs
    df_resultado["cura_segun_ODE"] = curacion_segun_ode
    df_resultado["tratamiento_recomendado"] = recomendaciones
    df_resultado["archivo_tratamiento"] = archivos_tratamiento
    df_resultado.to_csv(output_csv, index=False)
    print(f"\nResultados guardados en {output_csv}")

    # Resumen global en consola
    print("\nRESUMEN GLOBAL")
    print(f"Tau usado: {args.tau}")
    print(f"Promedio delta_log: {promedio_delta:.4f}")
    print(f"Promedio log_mdr: {promedio_log_mdr:.4f}")
    print(f"Porcentaje de cura: {porcentaje_cura:.2f}%")
    print(f"Tiempo total: {tiempo_total:.2f} segundos ({tiempo_total/60:.2f} minutos)")
    print(f"Probabilidad de cura media (clasificador): {prob_cura.mean():.4f}")
    print(f"% tratamiento recomendado por clasificador: {(recomendacion_cura == 'Tratamiento recomendado').mean()*100:.1f}%")

    # Generar gráficas para el paciente CURADO
    
    if paciente_cura_idx is not None:
        print(f"\n=== Generando gráficas para paciente CURADO (índice {paciente_cura_idx}) ===")
        generar_graficas_paciente(
            paciente_cura_static_tensor, paciente_cura_static_dict,
            paciente_cura_top_candidatos, args.output_dir,
            ode_model, stats, device, args.tau,
            paciente_id=paciente_cura_idx, sufijo="_cura"
        )
    else:
        print("\nNo se encontró ningún paciente CURADO en la muestra.")

    # Generar gráficas para el paciente NO CURADO
 
    if paciente_no_curado_idx is not None:
        print(f"\nGenerando gráficas para paciente NO CURADO (índice {paciente_no_curado_idx}) ===")
        generar_graficas_paciente(
            paciente_no_curado_static_tensor, paciente_no_curado_static_dict,
            paciente_no_curado_top_candidatos, args.output_dir,
            ode_model, stats, device, args.tau,
            paciente_id=paciente_no_curado_idx, sufijo="_no_curado"
        )
    else:
        print("\nNo se encontró ningún paciente NO CURADO en la muestra.")

    # Opcional: gráficas del primer paciente (sin sufijo, por compatibilidad)
    
    if top_candidatos_primer_paciente:
        print(f"\nGenerando gráficas para el primer paciente (índice 0, sin sufijo)")
        generar_graficas_paciente(
            static_primer_paciente, static_dict_primer_paciente,
            top_candidatos_primer_paciente, args.output_dir,
            ode_model, stats, device, args.tau,
            paciente_id=0, sufijo=""
        )

if __name__ == "__main__":
    main()