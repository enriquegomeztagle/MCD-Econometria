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

df_raw = pd.read_excel(DATA_PATH, header=0, sheet_name="tabla 4")

col_map = {_normalize_col(c): c for c in df_raw.columns}


def pick(*cands):
    for c in cands:
        key = _normalize_col(c)
        if key in col_map:
            return col_map[key]
    raise KeyError(
        f"No se encontró ninguna de las columnas: {cands}. Disponibles: {list(df_raw.columns)}"
    )


col_ahorro = pick("ahorro", "ahorro mdpcop", "ahorro (mdpcop)")
col_inv = pick("inversion", "inversión", "inversion mdpcop", "inversion (mdpcop)")


_df = df_raw[[col_ahorro, col_inv]].rename(
    columns={col_ahorro: "ahorro", col_inv: "inversion"}
)
for c in ["ahorro", "inversion"]:
    _df[c] = _df[c].apply(to_number)

df = _df.dropna(subset=["ahorro", "inversion"]).copy()
df = df[df["ahorro"] > 0]


df["inv_ahorro"] = 1.0 / df["ahorro"]

plot_dir = os.path.normpath(os.path.join(HERE, "../../plots/python/q4/"))
os.makedirs(plot_dir, exist_ok=True)

plt.figure()
plt.scatter(df["ahorro"], df["inversion"])
plt.xlabel("Ahorro (mdp COP)")
plt.ylabel("Inversión (mdp COP)")
plt.title("Dispersión: Inversión vs Ahorro (mdp COP)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "scatter_inversion_vs_ahorro_mdpcop.png"), dpi=150)
plt.close()


plt.figure()
plt.scatter(df["inv_ahorro"], df["inversion"])
plt.xlabel("1 / Ahorro")
plt.ylabel("Inversión (mdp COP)")
plt.title("Dispersión: Inversión vs 1/Ahorro (modelo recíproco)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "scatter_inversion_vs_inv_ahorro.png"), dpi=150)
plt.close()

X = sm.add_constant(df["inv_ahorro"])
I = df["inversion"]
model = sm.OLS(I, X, missing="drop").fit()

alpha = float(model.params.get("const", model.params.iloc[0]))
beta = float(model.params.get("inv_ahorro", model.params.iloc[1]))

print("\n================  PREGUNTA 1: Modelo Recíproco (Colombia)  ================")
print("Modelo estimado: I = a + b*(1/S)")
print(f"a (asíntota cuando S→∞): {alpha:.4f}")
print(f"b: {beta:.4f}")
print(f"R^2: {model.rsquared:.4f}")
print(f"n: {int(model.nobs)}")


from math import isfinite

try:
    from scipy import stats as sps
except Exception:
    sps = None

if "const" in model.bse.index:
    se_alpha = float(model.bse["const"])
else:
    se_alpha = float(model.bse.iloc[0])

mu0 = 60000.0
if se_alpha == 0 or not isfinite(se_alpha):
    t_stat = float("nan")
    p_left = float("nan")
else:
    t_stat = (alpha - mu0) / se_alpha
    df_res = int(round(model.df_resid))
    if sps is not None:
        p_left = sps.t.cdf(t_stat, df=df_res)
    else:
        import math
        from math import erf

        p_left = 0.5 * (1 + erf(t_stat / math.sqrt(2)))

alpha_sig = 0.05
crit = None
if sps is not None and isfinite(t_stat):
    crit = sps.t.ppf(alpha_sig, df=df_res)

decision = (
    "rechazar Ho"
    if (isfinite(t_stat) and t_stat < (crit if crit is not None else -1.645))
    else "No rechazar Ho"
)

print("\nPrueba: H0: a ≥ 60000  vs  H1: a < 60000 (α=0.05)")
print(f"t = {t_stat:.4f}")
print(f"p-valor (cola izq.) = {p_left:.4g}")
print(f"Decisión: {decision}")

S_bar = float(df["ahorro"].mean())
I_bar = float(df["inversion"].mean())
E_bar_obs = -beta / (S_bar * I_bar)

I_hat_bar = alpha + beta * (1.0 / S_bar)
E_bar_hat = -beta / (S_bar * I_hat_bar)

print("\n================  Elasticidad – Modelo Recíproco  ================")
print(f"Elasticidad (evaluada en S̄={S_bar:.2f}, ȳ={I_bar:.2f}): {E_bar_obs:.4f}")
print(
    f"Interpretación: cerca de S̄, un +1% en el ahorro cambia la inversión en ≈{E_bar_obs*100:.2f}%."
)
