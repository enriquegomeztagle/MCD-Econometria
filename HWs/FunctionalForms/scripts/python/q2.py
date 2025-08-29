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

df_raw = pd.read_excel(DATA_PATH, header=0, sheet_name="tabla 2")

col_map = {_normalize_col(c): c for c in df_raw.columns}


def pick(*cands):
    for c in cands:
        normalized_c = _normalize_col(c)
        if normalized_c in col_map:
            return col_map[normalized_c]
    raise KeyError(
        f"No se encontró ninguna de las columnas candidatas: {cands} en {list(df_raw.columns)}"
    )


col_ahorro = pick("ahorro")
col_inv = pick("inversion", "inversión", "inversión EUR")

_df = df_raw[[col_ahorro, col_inv]].rename(
    columns={col_ahorro: "ahorro", col_inv: "inversion"}
)
for c in ["ahorro", "inversion"]:
    _df[c] = _df[c].apply(to_number)

df = _df.copy()
df_log = df[df["ahorro"] > 0].copy()

OUT_DIR = os.path.normpath(os.path.join(HERE, "../../plots/python/q2/"))
os.makedirs(OUT_DIR, exist_ok=True)

plt.figure()
plt.scatter(df["ahorro"], df["inversion"])
plt.xlabel("Ahorro (mde)")
plt.ylabel("Inversión (mde)")
plt.title("Dispersión: Inversión vs Ahorro (mde)")
plt.grid(True, linestyle="--", alpha=0.4)
orig_plot_path = os.path.join(OUT_DIR, "dispersion_inversion_vs_ahorro_mde.png")
plt.tight_layout()
plt.savefig(orig_plot_path, dpi=150)
plt.close()

plt.figure()
plt.scatter(np.log(df_log["ahorro"]), df_log["inversion"])
plt.xlabel("ln(Ahorro)")
plt.ylabel("Inversión (mde)")
plt.title("Dispersión lin–log: Inversión vs ln(Ahorro)")
plt.grid(True, linestyle="--", alpha=0.4)
log_plot_path = os.path.join(OUT_DIR, "dispersion_linlog_inversion_vs_ln_ahorro.png")
plt.tight_layout()
plt.savefig(log_plot_path, dpi=150)
plt.close()

lnS = np.log(df_log["ahorro"])
I = df_log["inversion"]
X = sm.add_constant(lnS)
model = sm.OLS(I, X, missing="drop").fit()

alpha = model.params.get("const", model.params.iloc[0])
beta = model.params.get("ahorro", model.params.iloc[1])

ci_alpha = (
    model.conf_int().loc["const"].tolist()
    if "const" in model.conf_int().index
    else list(model.conf_int().iloc[0])
)

print(
    "\n================  PREGUNTA 1 – Rendimientos Decrecientes (lin–log)  ================"
)
print("Modelo: I = a + b ln(S)")
print(f"Ordenada (a): {alpha:.2f}")
print(f"Pendiente (b): {beta:.4f}")
print(f"IC 95% para a: [{ci_alpha[0]:.2f}, {ci_alpha[1]:.2f}]")
print(f"R^2: {model.rsquared:.4f}")

I_bar = float(df_log["inversion"].mean())
elasticidad = beta / I_bar

print(
    "\n================  PREGUNTA 2 – Modelo Rendimientos Decrecientes  ================"
)
print(
    f"Elasticidad estimada (criterio examen, b/ȳ con ȳ={I_bar:.2f}): {elasticidad:.4f}"
)
print(
    f"Interpretación: en promedio, un +1% en S se asocia con ≈{elasticidad*100:.2f}% en I."
)

try:
    from scipy import stats as sps
except Exception:
    import math

    sps = None

if "ahorro" in model.params.index:
    se_beta = float(model.bse["ahorro"])
else:
    se_beta = float(model.bse.iloc[1])

b_hat = float(beta)
se_b = se_beta

t_stat = (b_hat - 100.0) / se_b
df_res = int(round(model.df_resid))

if sps is not None:
    pval_right = sps.t.sf(t_stat, df=df_res)
else:
    from math import erf, sqrt

    phi = 0.5 * (1.0 + erf(t_stat / sqrt(2)))
    pval_right = 1.0 - phi

alpha_sig = 0.05
decision_right = "rechazar H0" if pval_right < alpha_sig else "No rechazar H0"

print(
    "\n================  PREGUNTA 3 – Afirmación sobre incremento del 1% (Francia)  ================"
)
print("Hipótesis: H0: b <= 100  vs  H1: b > 100")
print(f"b_hat = {b_hat:.4f}, se(b) = {se_b:.4f}, df = {df_res}")
print(f"Estadístico t = {t_stat:.4f}")
print(f"p-valor (cola derecha) = {pval_right:.4g}")
impacto_1pct = 0.01 * b_hat
print(f"Impacto de 1% en S: 0.01*b = {impacto_1pct:.4f} mde")
print(f"Decisión (α=0.05): {decision_right}")
