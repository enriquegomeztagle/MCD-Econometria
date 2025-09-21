# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import spearmanr

import os
from pathlib import Path

# %%
DATA_DIR = Path("../../../data/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LATEX_OUT = Path("../../../docs/latex_utils/tables")
LATEX_OUT.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = Path("../../../plots/python/ex3/")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# %%
def save_plot(
    plot: plt.Figure,
    filename: str,
    format: str = "png",
    dpi: int = 300,
    close: bool = True,
):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PLOTS_DIR / f"{filename}.{format}"
    try:
        plot.savefig(filepath, format=format, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(plot)
        print(
            f"\nPlot {filename}.{format} saved correctly in {PLOTS_DIR}/{filename}.{format}"
        )
    except Exception as e:
        print(f"\nCould not save plot {filename}.{format}. Reason: {e}")


# %%
def save_latex_table(df, filename: str, rename_map: dict, caption: str, label: str):
    try:
        table_tex = df.rename(columns=rename_map).to_latex(
            index=False,
            float_format="%.4f",
            caption=caption,
            label=label,
        )
        with open(LATEX_OUT / filename, "w") as f:
            f.write(table_tex)
        print(f"\nFile {filename} exported correctly in {LATEX_OUT}/{filename}")
    except Exception as e:
        print(f"\nCould not export {filename}. Reason: {e}")


# %%
df = pd.read_excel(os.path.join(DATA_DIR, "Table2_8.xls"))
print(df.head())

# %%
cols = [c.strip() for c in df.columns]
df.columns = cols

food_candidates = [
    c
    for c in cols
    if c.strip().lower()
    in [
        "food",
        "foodexp",
        "gasto_alimentos",
        "gasto_alimentario",
        "y",
        "food expenditure",
    ]
]
total_candidates = [
    c
    for c in cols
    if c.strip().lower()
    in ["total", "totalexp", "gasto_total", "x", "total expenditure"]
]

if not food_candidates or not total_candidates:
    num_cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    if len(num_cols) < 2:
        tmp = df.copy()
        for c in cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        num_cols = [c for c in cols if np.issubdtype(tmp[c].dtype, np.number)]
    food_col = food_candidates[0] if food_candidates else num_cols[0]
    total_col = total_candidates[0] if total_candidates else num_cols[1]
else:
    food_col = food_candidates[0]
    total_col = total_candidates[0]

T28 = df[[food_col, total_col]].copy()
T28.columns = ["Food", "Total"]

for c in ["Food", "Total"]:
    T28[c] = T28[c].astype(str).str.strip()
    T28[c] = T28[c].str.replace(",", ".", regex=False)
    T28[c] = T28[c].str.replace("%", "", regex=False)
    T28[c] = T28[c].str.replace(r"[^0-9.\-]", "", regex=True)
    T28[c] = T28[c].str.replace(r"\.$", "", regex=True)
    T28[c] = pd.to_numeric(T28[c], errors="coerce")

before_n = len(T28)
T28 = T28.dropna().reset_index(drop=True)
after_n = len(T28)
if after_n < before_n:
    print(
        f"Advertencia: se eliminaron {before_n-after_n} fila(s) por limpieza numérica en Tabla 2.8."
    )

# %%
model_a = smf.ols("Food ~ Total", data=T28).fit()
print(model_a.summary())

coefs_a = (
    model_a.summary2()
    .tables[1]
    .reset_index()
    .rename(
        columns={
            "index": "Parametro",
            "Coef.": "Coef",
            "Std.Err.": "EE",
            "P>|t|": "pval",
        }
    )
)
coefs_a = coefs_a[["Parametro", "Coef", "EE", "t", "pval", "[0.025", "0.975]"]]
coefs_a.columns = ["Parámetro", "Coef", "EE", "t", "p-valor", "IC 2.5%", "IC 97.5%"]
save_latex_table(
    coefs_a,
    "q3_a_coefs.tex",
    {},
    "Pregunta 3(a): Coeficientes OLS de Food sobre Total",
    "tab:q3a_coefs",
)

fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.scatter(model_a.fittedvalues, model_a.resid)
ax1.axhline(0, linestyle="--")
ax1.set_xlabel("Ajustados")
ax1.set_ylabel("Residuos")
ax1.set_title("Q3(a): Residuos vs Ajustados (Food~Total)")
save_plot(fig1, "q3_a_resid_vs_fitted")

fig2 = plt.figure(figsize=(6, 4))
qqplot(model_a.resid, line="s", ax=plt.gca())
plt.title("Q3(a): QQ-plot de residuos (Food~Total)")
save_plot(fig2, "q3_a_resid_qqplot")

JB_stat, JB_pval, _, _ = jarque_bera(model_a.resid)
print(f"Jarque-Bera (a): stat={JB_stat:.4f}, pval={JB_pval:.4f}")

# %%
resid2 = model_a.resid**2
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.scatter(T28["Total"], resid2)
ax3.set_xlabel("Total")
ax3.set_ylabel("Residuos^2")
ax3.set_title("Q3(b): Residuos^2 vs Total")
save_plot(fig3, "q3_b_resid2_vs_total")

# %%
T28_park = T28.copy()
T28_park["ln_resid2"] = np.log(np.maximum(resid2, 1e-12))
T28_park["ln_Total"] = np.log(np.maximum(T28_park["Total"], 1e-12))
park_res = smf.ols("ln_resid2 ~ ln_Total", data=T28_park).fit()
print(park_res.summary())

park_tab = (
    park_res.summary2()
    .tables[1]
    .reset_index()
    .rename(
        columns={
            "index": "Parametro",
            "Coef.": "Coef",
            "Std.Err.": "EE",
            "P>|t|": "pval",
        }
    )
)
park_tab = park_tab[["Parametro", "Coef", "EE", "t", "pval", "[0.025", "0.975]"]]
park_tab.columns = ["Parámetro", "Coef", "EE", "t", "p-valor", "IC 2.5%", "IC 97.5%"]
save_latex_table(
    park_tab,
    "q3_c_park.tex",
    {},
    "Pregunta 3(c): Prueba de Park (ln(resid^2)~ln(Total))",
    "tab:q3c_park",
)

rho, rho_p = spearmanr(np.abs(model_a.resid), T28["Total"])
print(f"Spearman |resid| vs Total: rho={rho:.4f}, pval={rho_p:.4f}")
rho2, rho2_p = spearmanr(resid2, T28["Total"])
print(f"Spearman resid^2 vs Total: rho={rho2:.4f}, pval={rho2_p:.4f}")

spearman_df = pd.DataFrame(
    {
        "Métrica": ["|resid| vs Total", "resid^2 vs Total"],
        "rho": [rho, rho2],
        "p-valor": [rho_p, rho2_p],
    }
)
save_latex_table(
    spearman_df,
    "q3_c_spearman.tex",
    {},
    "Pregunta 3(c): Correlación de Spearman para heteroscedasticidad",
    "tab:q3c_spearman",
)

exog = sm.add_constant(T28[["Total"]])
W_stat, W_pval, fval_w, fpval_w = het_white(model_a.resid, exog)
white_df = pd.DataFrame(
    {"Prueba": ["White"], "Estadístico": [W_stat], "p-valor": [W_pval]}
)
save_latex_table(
    white_df,
    "q3_c_white.tex",
    {},
    "Pregunta 3(c): Prueba de White (modelo lineal)",
    "tab:q3c_white",
)

# %%
T28_log = T28.copy()
T28_log["ln_Food"] = np.log(np.maximum(T28_log["Food"], 1e-12))
T28_log["ln_Total"] = np.log(np.maximum(T28_log["Total"], 1e-12))
model_d = smf.ols("ln_Food ~ ln_Total", data=T28_log).fit()
print(model_d.summary())

coefs_d = (
    model_d.summary2()
    .tables[1]
    .reset_index()
    .rename(
        columns={
            "index": "Parametro",
            "Coef.": "Coef",
            "Std.Err.": "EE",
            "P>|t|": "pval",
        }
    )
)
coefs_d = coefs_d[["Parametro", "Coef", "EE", "t", "pval", "[0.025", "0.975]"]]
coefs_d.columns = ["Parámetro", "Coef", "EE", "t", "p-valor", "IC 2.5%", "IC 97.5%"]
save_latex_table(
    coefs_d,
    "q3_d_loglog_coefs.tex",
    {},
    "Pregunta 3(d): Coeficientes OLS (log-log)",
    "tab:q3d_loglog_coefs",
)

fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.scatter(model_d.fittedvalues, model_d.resid)
ax4.axhline(0, linestyle="--")
ax4.set_xlabel("Ajustados (log)")
ax4.set_ylabel("Residuos (log)")
ax4.set_title("Q3(d): Residuos vs Ajustados (log-log)")
save_plot(fig4, "q3_d_loglog_resid_vs_fitted")

fig5 = plt.figure(figsize=(6, 4))
qqplot(model_d.resid, line="s", ax=plt.gca())
plt.title("Q3(d): QQ-plot de residuos (log-log)")
save_plot(fig5, "q3_d_loglog_resid_qqplot")

exog_ll = sm.add_constant(T28_log[["ln_Total"]])
W_stat_ll, W_pval_ll, _, _ = het_white(model_d.resid, exog_ll)
BP_stat_ll, BP_pval_ll, _, _ = het_breuschpagan(model_d.resid, exog_ll)
hetero_ll = pd.DataFrame(
    {"Prueba": ["White", "Breusch-Pagan"], "p-valor": [W_pval_ll, BP_pval_ll]}
)
save_latex_table(
    hetero_ll,
    "q3_d_loglog_hetero.tex",
    {},
    "Pregunta 3(d): Pruebas de heterocedasticidad (log-log)",
    "tab:q3d_loglog_hetero",
)

print("\nListo Q3: Figuras en", PLOTS_DIR)
print("Tablas LaTeX en", LATEX_OUT)

# %%
