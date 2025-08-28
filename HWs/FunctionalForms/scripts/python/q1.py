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
    s = str(x)
    s = s.replace("$", "").replace(",", "")
    s = s.strip()
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

df_raw = pd.read_excel(DATA_PATH, header=1, sheet_name="tabla 1")
print(df_raw.head())

col_map = {_normalize_col(c): c for c in df_raw.columns}


def pick(*cands):
    for c in cands:
        if c in col_map:
            return col_map[c]
    raise KeyError(
        f"No se encontró ninguna de las columnas: {cands} en {list(df_raw.columns)}"
    )


col_ahorro = pick("ahorro")
col_inv = pick("inversion", "inversio n", "inversion ", "inversi on", "inversión")

df = df_raw[[col_ahorro, col_inv]].rename(
    columns={col_ahorro: "ahorro", col_inv: "inversion"}
)


for c in ["ahorro", "inversion"]:
    df[c] = df[c].apply(to_number)


df_log = df[(df["ahorro"] > 0) & (df["inversion"] > 0)].copy()

OUT_DIR = os.path.join("../../plots/python/q1/")
os.makedirs(OUT_DIR, exist_ok=True)

plt.figure()
plt.scatter(df["ahorro"], df["inversion"])
plt.xlabel("Ahorro (mil USD)")
plt.ylabel("Inversión (mil USD)")
plt.title("Dispersión: Inversión vs Ahorro (mil USD)")
plt.grid(True, linestyle="--", alpha=0.4)
orig_plot_path = os.path.join(OUT_DIR, "dispersion_inversion_vs_ahorro_mdd.png")
plt.tight_layout()
plt.savefig(orig_plot_path, dpi=150)
plt.close()

plt.figure()
plt.scatter(np.log(df_log["ahorro"]), np.log(df_log["inversion"]))
plt.xlabel("ln(Ahorro mil USD)")
plt.ylabel("ln(Inversión mil USD)")
plt.title("Dispersión log–log: ln(Inversión) vs ln(Ahorro)")
plt.grid(True, linestyle="--", alpha=0.4)
log_plot_path = os.path.join(OUT_DIR, "dispersion_loglog_inversion_vs_ahorro.png")
plt.tight_layout()
plt.savefig(log_plot_path, dpi=150)
plt.close()


lnS = np.log(df_log["ahorro"])
lnI = np.log(df_log["inversion"])
X = sm.add_constant(lnS)
model = sm.OLS(lnI, X, missing="drop").fit()

beta = model.params.get("ahorro", model.params.iloc[1])
alpha = model.params.get("const", model.params.iloc[0])

conf_int = model.conf_int().to_dict()
ci_beta = conf_int.get("ahorro") or list(model.conf_int().iloc[1])

print("\n================  PREGUNTA 1 – Elasticidad Constante  ================")
print(f"n (válidas para log-log): {len(df_log)}")
print(f"Modelo: ln(I) = a + b ln(S)")
print(f"Intercepto (a): {alpha:.6f}")
print(f"Elasticidad (b): {beta:.6f}")
print(f"IC 95% para b: [{ci_beta[0]:.6f}, {ci_beta[1]:.6f}]")
print(f"R^2: {model.rsquared:.4f}")


print("\nResumen del modelo OLS (log–log):\n")
print(model.summary())

print("\n================  PREGUNTA 2 – Intervalo de Confianza  ================")
print(
    f"Intervalo de confianza al 95% para la elasticidad: [{ci_beta[0]:.6f}, {ci_beta[1]:.6f}]"
)
print(
    f"Interpretación: Existe un 95% de confianza de que la elasticidad de la inversión respecto al ahorro se encuentra entre {ci_beta[0]:.4f} y {ci_beta[1]:.4f}. Esto indica que, en promedio, un aumento de 1% en el ahorro incrementa la inversión entre {ci_beta[0]*100:.2f}% y {ci_beta[1]*100:.2f}%."
)

mask_pos = (df["ahorro"] > 0) & (df["inversion"] > 0)
df_log_orig = df.loc[mask_pos, ["ahorro", "inversion"]].copy()
lnS_o = np.log(df_log_orig["ahorro"])
lnI_o = np.log(df_log_orig["inversion"])
X_o = sm.add_constant(lnS_o)
model_o = sm.OLS(lnI_o, X_o, missing="drop").fit()

S0_orig = 200.0
exog_names_o = model_o.model.exog_names
exog_200_o = pd.DataFrame([[1.0, np.log(S0_orig)]], columns=exog_names_o)
pred_200_o = model_o.get_prediction(exog_200_o)
frm_o = pred_200_o.summary_frame(alpha=0.05)

ln_pred_lo, ln_pred_hi = float(frm_o.loc[0, "obs_ci_lower"]), float(
    frm_o.loc[0, "obs_ci_upper"]
)
I_pred_lo = np.exp(ln_pred_lo)
I_pred_hi = np.exp(ln_pred_hi)

print("\n=====  PREGUNTA 3 – Formato examen (PRONÓSTICO)  =====")
print(
    f"IC 95% para I|S=200 en mdd (intervalo de pronóstico): [{I_pred_lo:,.2f}, {I_pred_hi:,.2f}]"
)
