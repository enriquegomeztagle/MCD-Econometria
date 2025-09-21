# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf


import os
from pathlib import Path

# %%
DATA_DIR = Path("../../../data/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LATEX_OUT = Path("../../../docs/latex_utils/tables")
LATEX_OUT.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = Path("../../../plots/python/ex7/")
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
df = pd.read_excel(os.path.join(DATA_DIR, "Table12_7.xls"))
print(df.head())

# %%
if "YEAR" in df.columns:
    df = df.sort_values("YEAR").reset_index(drop=True)

rename_cols = {
    "Year": "YEAR",
    "year": "YEAR",
    "C ": "C",
    "G ": "G",
    "I ": "I",
    "L ": "L",
    "H ": "H",
    "A ": "A",
}
df = df.rename(columns={k: v for k, v in rename_cols.items() if k in df.columns})

required = ["C", "I", "L", "H", "A"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas requeridas en el DataFrame: {missing}")

for c in required:
    df = df[df[c] > 0]

df = df.copy()
df["lnC"] = np.log(df["C"])
df["lnI"] = np.log(df["I"])
df["lnL"] = np.log(df["L"])
df["lnH"] = np.log(df["H"])
df["lnA"] = np.log(df["A"])

# %%
Y = df["lnC"]
X = df[["lnI", "lnL", "lnH", "lnA"]]
X = sm.add_constant(X)
model = sm.OLS(Y, X, missing="drop").fit()

print("\n================  Resultados de la Regresión (Log-Log)  ================")
print(model.summary())

# %%
coefs = model.params.rename("Coeficiente")
se = model.bse.rename("EE")
tvals = model.tvalues.rename("t")
pvals = model.pvalues.rename("p-valor")
resumen_df = (
    pd.concat([coefs, se, tvals, pvals], axis=1)
    .reset_index()
    .rename(columns={"index": "Parámetro"})
)
resumen_df.loc[len(resumen_df.index)] = ["R^2", model.rsquared, np.nan, np.nan, np.nan]
resumen_df.loc[len(resumen_df.index)] = [
    "R^2 ajustado",
    model.rsquared_adj,
    np.nan,
    np.nan,
    np.nan,
]
resumen_df.loc[len(resumen_df.index)] = ["N", int(model.nobs), np.nan, np.nan, np.nan]

save_latex_table(
    resumen_df,
    filename="q7_a.tex",
    rename_map={
        "Parámetro": "Parámetro",
        "Coeficiente": "Coef.",
        "EE": "EE",
        "t": "t",
        "p-valor": "p-valor",
    },
    caption=(
        r"Resultados OLS para $\ln C_t$ con regresores $\ln I_t, \ln L_t, \ln H_t, \ln A_t$. "
        r"Errores estándar entre paréntesis."
    ),
    label="tab:q7_a",
)

# %%
resid = model.resid
fitted = model.fittedvalues
influence = model.get_influence()
std_resid = influence.resid_studentized_internal

if "YEAR" not in df.columns:
    df["YEAR"] = np.arange(len(df))

df["resid"] = resid
df["std_resid"] = std_resid

dw_stat = sm.stats.stattools.durbin_watson(resid)
print(f"\nEstadístico Durbin–Watson: {dw_stat:.4f}")

# %%
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(df["YEAR"], df["resid"], marker="o", linestyle="-")
ax1.axhline(0, linestyle="--", linewidth=1)
ax1.set_title("Residuos vs. Año")
ax1.set_xlabel("Año")
ax1.set_ylabel("Residuo")
save_plot(fig1, "ex7_residuos_vs_time")

# %%
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(df["YEAR"], df["std_resid"], marker="o", linestyle="-")
ax2.axhline(0, linestyle="--", linewidth=1)
ax2.axhline(2, linestyle=":", linewidth=1)
ax2.axhline(-2, linestyle=":", linewidth=1)
ax2.set_title("Residuos estandarizados vs. Año")
ax2.set_xlabel("Año")
ax2.set_ylabel("Residuo estandarizado")
save_plot(fig2, "ex7_stdres_vs_time")

# %%
fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.scatter(fitted, resid)
ax3.axhline(0, linestyle="--", linewidth=1)
ax3.set_title("Residuos vs. Valores ajustados")
ax3.set_xlabel("Ajustados")
ax3.set_ylabel("Residuo")
save_plot(fig3, "ex7_residuos_vs_fitted")

# %%
fig4 = qqplot(resid, line="45", fit=True)
plt.title("Q-Q plot de residuos")
save_plot(fig4, "ex7_qqplot_residuos")

# %%
fig5, ax5 = plt.subplots(figsize=(8, 4))
plot_acf(resid, ax=ax5, lags=min(20, len(df) - 2))
ax5.set_title("ACF de los residuos")
save_plot(fig5, "ex7_acf_residuos")

# %%
resid_lag = resid.shift(1)
fig6, ax6 = plt.subplots(figsize=(6, 5))
ax6.scatter(resid_lag[1:], resid[1:])
ax6.set_title(r"Dispersión $u_t$ vs $u_{t-1}$")
ax6.set_xlabel(r"$u_{t-1}$")
ax6.set_ylabel(r"$u_t$")
valid = (~resid_lag.isna()) & (~resid.isna())
if valid.sum() > 1:
    b1, b0, r, p, se = stats.linregress(resid_lag[valid], resid[valid])
    xgrid = np.linspace(resid_lag[valid].min(), resid_lag[valid].max(), 100)
    ax6.plot(xgrid, b0 + b1 * xgrid, linestyle="--")
save_plot(fig6, "ex7_scatter_resid_lag")
