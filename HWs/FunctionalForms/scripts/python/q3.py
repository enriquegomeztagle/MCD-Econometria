import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def _normalize_col(s: str) -> str:
    s = s.strip().lower()
    reemplazos = str.maketrans(
        {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "ä": "a",
            "ë": "e",
            "ï": "i",
            "ö": "o",
            "ü": "u",
            "ñ": "n",
        }
    )
    return s.translate(reemplazos)


def to_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).replace("$", "").replace(",", "").strip()
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else np.nan


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(HERE, "../../data/tablaTareaFF.xlsx"))
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró el archivo de datos en: {DATA_PATH}")

df_raw = pd.read_excel(DATA_PATH, header=0, sheet_name="tabla 3")

col_map = {_normalize_col(c): c for c in df_raw.columns}


def pick_col(*cands):
    for c in cands:
        key = _normalize_col(c)
        if key in col_map:
            return col_map[key]
    raise KeyError(
        f"No encontré ninguna de las columnas {cands}. Disponibles: {list(df_raw.columns)}"
    )


col_ahorro = pick_col("ahorro", "ahorro (mdp mxn)", "ahorro mdp mxn", "ahorro mdpmxn")
col_inv = pick_col(
    "inversion mxn",
    "inversion (mdp mxn)",
    "inversion mdp mxn",
    "inversion",
    "inversión mxn",
    "inversión (mdp mxn)",
    "inversion mxn",
)

_df = df_raw[[col_ahorro, col_inv]].rename(
    columns={col_ahorro: "ahorro", col_inv: "inversion"}
)
for c in ["ahorro", "inversion"]:
    _df[c] = _df[c].apply(to_number)

df = _df.dropna(subset=["inversion", "ahorro"]).copy()
df = df[(df["inversion"] > 0)]

plt.figure(figsize=(8, 6))
plt.scatter(df["ahorro"], df["inversion"], color="blue", alpha=0.7)
plt.xlabel("Ahorro (mdp MXN)")
plt.ylabel("Inversión (mdp MXN)")
plt.title("Escala original: Inversión vs Ahorro")
plt.grid(True)
plot_dir = os.path.normpath(os.path.join(HERE, "../../plots/python/q3/"))
os.makedirs(plot_dir, exist_ok=True)
plt.savefig(os.path.join(plot_dir, "scatter_inversion_ahorro.png"))
plt.close()

df["ln_inversion"] = np.log(df["inversion"])
plt.figure(figsize=(8, 6))
plt.scatter(df["ahorro"], df["ln_inversion"], color="green", alpha=0.7)
plt.xlabel("Ahorro (mdp MXN)")
plt.ylabel("ln(Inversión)")
plt.title("Transformación log–lin: ln(Inversión) vs Ahorro")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "scatter_ln_inversion_ahorro.png"))
plt.close()

X = sm.add_constant(df["ahorro"])
y = df["ln_inversion"]
model = sm.OLS(y, X).fit()

a = model.params["const"]
b = model.params["ahorro"]

print("Modelo: ln(Inversión) = a + b * Ahorro")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")

exog_names = model.model.exog_names
exog = pd.DataFrame([[1.0, 1000.0]], columns=exog_names)
pred = model.get_prediction(exog)
pred_summary = pred.summary_frame(alpha=0.05)

ci_lower_pred = np.exp(pred_summary["obs_ci_lower"][0])
ci_upper_pred = np.exp(pred_summary["obs_ci_upper"][0])

ci_lo4 = float(f"{ci_lower_pred:.4f}")
ci_hi4 = float(f"{ci_upper_pred:.4f}")

print(
    "\n================  PREGUNTA 1: Intervalo de pronóstico 95% para Inversión cuando Ahorro=1000  ================"
)
print(
    f"Intervalo de PRONÓSTICO 95% para Inversión cuando Ahorro=1000: [{ci_lo4:,.4f},{ci_hi4:,.4f}]"
)

# ===========================
# PREGUNTA 2: Elasticidad – evaluar en media(S) y sugerir inciso
# ===========================
S_mean = float(df["ahorro"].mean())
E_mean = b * S_mean
print(
    "\n================  PREGUNTA 2: Elasticidad – Modelo de Crecimiento Exponencial  ================"
)
print(
    f"Elasticidad estimada de I respecto a S en S=media(S)={S_mean:.2f}: {E_mean:.4f}"
)
print(
    f"Interpretación: cerca del ahorro promedio, un +1% en S se asocia con ≈{E_mean:.2f}% en I."
)
