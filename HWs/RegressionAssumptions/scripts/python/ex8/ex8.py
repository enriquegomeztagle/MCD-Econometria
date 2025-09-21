# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset


import os
from pathlib import Path

# %%
DATA_DIR = Path("../../../data/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LATEX_OUT = Path("../../../docs/latex_utils/tables")
LATEX_OUT.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = Path("../../../plots/python/ex8/")
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
df = pd.read_excel(os.path.join(DATA_DIR, "Table12_9.xls"))
print(df.head())

# %%
cols_map = {
    "Year": "year",
    "Sales": "sales",
    "Inventories": "inventories",
    "Ratio": "ratio",
}
df = df.rename(columns=cols_map)

numeric_cols = ["year", "sales", "inventories", "ratio"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df = df.sort_values("year").reset_index(drop=True)

print("\nDimensiones:", df.shape)
print("\nDescripción estadística básica:\n", df.describe().T)

# %%
desc = (
    df[["sales", "inventories", "ratio"]]
    .describe()
    .T.reset_index()
    .rename(columns={"index": "variable"})
)
save_latex_table(
    desc,
    filename="ex8_descriptivos.tex",
    rename_map={
        "variable": "Variable",
        "count": "N",
        "mean": "Media",
        "std": "Desv.Est.",
        "min": "Mínimo",
        "25%": "P25",
        "50%": "P50",
        "75%": "P75",
        "max": "Máximo",
    },
    caption="Estadísticos descriptivos de ventas, inventarios y razón inventarios/ventas.",
    label="tab:ex8_desc",
)

# %%
corr = (
    df[["sales", "inventories", "ratio"]]
    .corr()
    .reset_index()
    .rename(columns={"index": "variable"})
)
save_latex_table(
    corr,
    filename="ex8_correlaciones.tex",
    rename_map={
        "variable": "Variable",
        "sales": "Ventas",
        "inventories": "Inventarios",
        "ratio": "Razón",
    },
    caption="Matriz de correlaciones.",
    label="tab:ex8_corr",
)

# %%
plt.figure()
plt.plot(df["year"], df["sales"], marker="o", label="Ventas")
plt.plot(df["year"], df["inventories"], marker="s", label="Inventarios")
plt.xlabel("Año")
plt.ylabel("Nivel")
plt.title("Series en el tiempo: Ventas e Inventarios")
plt.legend()
save_plot(plt.gcf(), filename="ex8_series_ventas_inventarios")

# %%
plt.figure()
plt.plot(df["year"], df["ratio"], marker="o")
plt.xlabel("Año")
plt.ylabel("Inventarios/Ventas")
plt.title("Razón Inventarios/Ventas en el tiempo")
save_plot(plt.gcf(), filename="ex8_series_ratio")

# %%
plt.figure()
plt.scatter(df["sales"], df["inventories"])
b1, b0, r, p, se = stats.linregress(df["sales"], df["inventories"])
xline = np.linspace(df["sales"].min(), df["sales"].max(), 100)
plt.plot(xline, b0 + b1 * xline)
plt.xlabel("Ventas")
plt.ylabel("Inventarios")
plt.title("Inventarios vs. Ventas (con línea OLS)")
save_plot(plt.gcf(), filename="ex8_scatter_inv_vs_sales")

# %%
X1 = sm.add_constant(df[["sales"]])
model1 = sm.OLS(df["inventories"], X1).fit()
print("\n=== Modelo 1: Inventarios ~ Ventas ===\n", model1.summary())

# %%
X2 = sm.add_constant(df[["sales", "year"]])
model2 = sm.OLS(df["inventories"], X2).fit()
print("\n=== Modelo 2: Inventarios ~ Ventas + Año ===\n", model2.summary())


# %%
def _params_to_df(res):
    out = (
        res.params.to_frame("coef")
        .join(res.bse.to_frame("se"))
        .join(res.tvalues.to_frame("t"))
        .join(res.pvalues.to_frame("p"))
    )
    ci = res.conf_int()
    ci.columns = ["ci_low", "ci_high"]
    out = out.join(ci)
    out.index.name = "Parametro"
    return out.reset_index()


# %%
coef1 = _params_to_df(model1)
coef2 = _params_to_df(model2)

save_latex_table(
    coef1,
    filename="ex8_ols_m1.tex",
    rename_map={
        "Parametro": "Parámetro",
        "coef": "Coef.",
        "se": "E.E.",
        "t": "t",
        "p": "p-valor",
        "ci_low": "IC 2.5%",
        "ci_high": "IC 97.5%",
    },
    caption="Resultados OLS: Inventarios sobre Ventas.",
    label="tab:ex8_m1",
)

# %%
save_latex_table(
    coef2,
    filename="ex8_ols_m2.tex",
    rename_map={
        "Parametro": "Parámetro",
        "coef": "Coef.",
        "se": "E.E.",
        "t": "t",
        "p": "p-valor",
        "ci_low": "IC 2.5%",
        "ci_high": "IC 97.5%",
    },
    caption="Resultados OLS: Inventarios sobre Ventas y Año.",
    label="tab:ex8_m2",
)

# %%
res1 = model1.resid
res2 = model2.resid

fig = qqplot(res1, line="45")
plt.title("Q-Q plot residuos Modelo 1")
save_plot(fig.figure, filename="ex8_qq_m1")

# %%
fig = qqplot(res2, line="45")
plt.title("Q-Q plot residuos Modelo 2")
save_plot(fig.figure, filename="ex8_qq_m2")

# %%
fig = plot_acf(res1, lags=12)
plt.title("ACF residuos Modelo 1")
save_plot(fig.figure, filename="ex8_acf_m1")

# %%
fig = plot_acf(res2, lags=12)
plt.title("ACF residuos Modelo 2")
save_plot(fig.figure, filename="ex8_acf_m2")

# %%
dw1 = sm.stats.stattools.durbin_watson(res1)
dw2 = sm.stats.stattools.durbin_watson(res2)
print(f"\nDurbin-Watson M1: {dw1:.4f}  |  M2: {dw2:.4f}")

# %%
bp1 = het_breuschpagan(res1, X1)  # (LM stat, LM pvalue, F stat, F pvalue)
bp2 = het_breuschpagan(res2, X2)
print("\nBreusch-Pagan M1 (LM, p, F, p):", tuple(round(x, 4) for x in bp1))
print("Breusch-Pagan M2 (LM, p, F, p):", tuple(round(x, 4) for x in bp2))

# %%
wt1 = het_white(res1, X1)
wt2 = het_white(res2, X2)
print("\nWhite test M1 (LM, p, F, p):", tuple(round(x, 4) for x in wt1))
print("White test M2 (LM, p, F, p):", tuple(round(x, 4) for x in wt2))

# %%
jb1 = sm.stats.stattools.jarque_bera(res1)
jb2 = sm.stats.stattools.jarque_bera(res2)
print("\nJarque-Bera M1 (JB, p, skew, kurt):", tuple(round(float(x), 4) for x in jb1))
print("Jarque-Bera M2 (JB, p, skew, kurt):", tuple(round(float(x), 4) for x in jb2))

# %%
reset1 = linear_reset(model1, use_f=True)
reset2 = linear_reset(model2, use_f=True)
print(
    "\nRESET Ramsey M1 (F, p, gl):",
    round(reset1.fvalue, 4),
    round(reset1.pvalue, 4),
    reset1.df_denom,
)
print(
    "RESET Ramsey M2 (F, p, gl):",
    round(reset2.fvalue, 4),
    round(reset2.pvalue, 4),
    reset2.df_denom,
)

# %%
idx = ["M1: Inv ~ Ventas", "M2: Inv ~ Ventas + Año"]
results_tests = (
    pd.DataFrame(
        {
            "Durbin-Watson": [dw1, dw2],
            "BP p-valor": [bp1[1], bp2[1]],
            "White p-valor": [wt1[1], wt2[1]],
            "JB p-valor": [float(jb1[1]), float(jb2[1])],
            "RESET p-valor": [reset1.pvalue, reset2.pvalue],
            "R2": [model1.rsquared, model2.rsquared],
            "R2 Ajustado": [model1.rsquared_adj, model2.rsquared_adj],
            "N": [int(model1.nobs), int(model2.nobs)],
        },
        index=idx,
    )
    .reset_index()
    .rename(columns={"index": "Modelo"})
)

save_latex_table(
    results_tests.round(4),
    filename="ex8_diagnosticos.tex",
    rename_map={
        "Modelo": "Modelo",
        "Durbin-Watson": "Durbin–Watson",
        "BP p-valor": "Breusch–Pagan p",
        "White p-valor": "White p",
        "JB p-valor": "Jarque–Bera p",
        "RESET p-valor": "RESET p",
        "R2": "$R^2$",
        "R2 Ajustado": "$R^2$ Ajustado",
        "N": "N",
    },
    caption="Pruebas de supuestos y métricas de ajuste para los modelos estimados.",
    label="tab:ex8_diagnosticos",
)

# %%
plt.figure()
plt.plot(df["year"], res1, marker="o")
plt.axhline(0, linestyle="--")
plt.xlabel("Año")
plt.ylabel("Residuo")
plt.title("Residuos en el tiempo – Modelo 1")
save_plot(plt.gcf(), filename="ex8_residuos_m1")

# %%
plt.figure()
plt.plot(df["year"], res2, marker="o")
plt.axhline(0, linestyle="--")
plt.xlabel("Año")
plt.ylabel("Residuo")
plt.title("Residuos en el tiempo – Modelo 2")
save_plot(plt.gcf(), filename="ex8_residuos_m2")

# %%
