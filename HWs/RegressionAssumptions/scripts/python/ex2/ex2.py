# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.stattools import jarque_bera

import os
from pathlib import Path

# %%
DATA_DIR = Path("../../../data/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LATEX_OUT = Path("../../../docs/latex_utils/tables")
LATEX_OUT.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = Path("../../../plots/python/ex2/")
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
df = pd.read_excel(os.path.join(DATA_DIR, "Table11_9.xls"))
print(df.head())

# %%
cols = [c.strip() for c in df.columns]
df.columns = cols
col_country_candidates = [c for c in cols if c.lower() in ["country", "pais", "país"]]
col_y_candidates = [
    c
    for c in cols
    if c.strip().upper() in ["Y", "STOCKS", "ACCIONES", "PRECIO_ACCIONES"]
]
col_x_candidates = [
    c
    for c in cols
    if c.strip().upper() in ["X", "CPI", "CONSUMER", "PRECIOS_CONSUMIDOR"]
]

# %%
if not col_country_candidates:
    if df.dtypes.iloc[0] == "object":
        col_country = cols[0]
    else:
        col_country = "Country"
        df[col_country] = [f"C{i+1}" for i in range(len(df))]
else:
    col_country = col_country_candidates[0]

# %%
if not col_y_candidates or not col_x_candidates:
    num_cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    if len(num_cols) < 2:
        raise ValueError(
            "No se encontraron suficientes columnas numéricas para Y y X en Table11_9.xlsx"
        )
    col_y = num_cols[0] if not col_y_candidates else col_y_candidates[0]
    col_x = num_cols[1] if not col_x_candidates else col_x_candidates[0]
else:
    col_y = col_y_candidates[0]
    col_x = col_x_candidates[0]

# %%
T119 = df[[col_country, col_y, col_x]].copy()
T119.columns = ["Country", "Y", "X"]

# %%
for c in ["Y", "X"]:
    T119[c] = T119[c].astype(str).str.strip()
    T119[c] = T119[c].str.replace(",", ".", regex=False)
    T119[c] = T119[c].str.replace("%", "", regex=False)
    T119[c] = T119[c].str.replace(r"[^0-9.\-]", "", regex=True)
    T119[c] = T119[c].str.replace(r"\.$", "", regex=True)
    T119[c] = pd.to_numeric(T119[c], errors="coerce")

# %%
before_rows = len(T119)
T119 = T119.dropna(subset=["Y", "X"]).reset_index(drop=True)
after_rows = len(T119)
if after_rows < before_rows:
    print(
        f"Advertencia: se eliminaron {before_rows - after_rows} fila(s) por valores no numéricos en X/Y tras limpieza."
    )

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(T119["X"], T119["Y"])
ax.set_xlabel("X: Δ% Precios al consumidor (anual)")
ax.set_ylabel("Y: Δ% Precios de acciones (anual)")
ax.set_title("Tabla 11.9: Dispersión Y vs X (20 países)")

mask_chile = T119["Country"].str.lower().str.contains("chile", na=False)
if mask_chile.any():
    row_chile = T119[mask_chile].iloc[0]
    ax.annotate(
        "Chile",
        (row_chile["X"], row_chile["Y"]),
        xytext=(5, 5),
        textcoords="offset points",
    )

save_plot(fig, filename="q2_scatter_t119")

# %%
model_b = smf.ols("Y ~ X", data=T119).fit()
print(model_b.summary())

coef_b = (
    model_b.summary2()
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
coef_b = coef_b[["Parametro", "Coef", "EE", "t", "pval", "[0.025", "0.975]"]]
coef_b.columns = ["Parámetro", "Coef", "EE", "t", "p-valor", "IC 2.5%", "IC 97.5%"]

save_latex_table(
    coef_b,
    filename="q2_b_coefs.tex",
    rename_map={},
    caption="Pregunta 2(b): Coeficientes OLS para Y~X (todos los países)",
    label="tab:q2b_coefs",
)

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.scatter(model_b.fittedvalues, model_b.resid)
ax2.axhline(0, linestyle="--")
ax2.set_xlabel("Valores ajustados")
ax2.set_ylabel("Residuos")
ax2.set_title("Q2(b): Residuos vs Ajustados (Y~X)")
save_plot(fig2, filename="q2_b_resid_vs_fitted")

fig3 = plt.figure(figsize=(6, 4))
qqplot(model_b.resid, line="s", ax=plt.gca())
plt.title("Q2(b): QQ-plot de residuos (Y~X)")
save_plot(fig3, filename="q2_b_resid_qqplot")

exog = sm.add_constant(T119[["X"]])
BP_stat, BP_pval, fval, fpval = het_breuschpagan(model_b.resid, exog)
W_stat, W_pval, fval_w, fpval_w = het_white(model_b.resid, exog)

het_table_b = pd.DataFrame(
    {
        "Prueba": ["Breusch-Pagan", "White"],
        "Estadístico": [BP_stat, W_stat],
        "p-valor": [BP_pval, W_pval],
    }
)

save_latex_table(
    het_table_b,
    filename="q2_b_hetero_tests.tex",
    rename_map={},
    caption="Pregunta 2(b): Pruebas de heterocedasticidad para Y~X (todos)",
    label="tab:q2b_hetero",
)

JB_stat, JB_pval, _, _ = jarque_bera(model_b.resid)
print(f"Jarque-Bera: stat={JB_stat:.4f}, pval={JB_pval:.4f}")

# %%
if mask_chile.any():
    T119_no_chile = T119.loc[~mask_chile].reset_index(drop=True)
else:
    T119_no_chile = T119.copy()

model_c = smf.ols("Y ~ X", data=T119_no_chile).fit()
print(model_c.summary())

coef_c = (
    model_c.summary2()
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
coef_c = coef_c[["Parametro", "Coef", "EE", "t", "pval", "[0.025", "0.975]"]]
coef_c.columns = ["Parámetro", "Coef", "EE", "t", "p-valor", "IC 2.5%", "IC 97.5%"]

save_latex_table(
    coef_c,
    filename="q2_c_coefs_no_chile.tex",
    rename_map={},
    caption="Pregunta 2(c): Coeficientes OLS para Y~X (sin Chile)",
    label="tab:q2c_coefs",
)

fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.scatter(model_c.fittedvalues, model_c.resid)
ax4.axhline(0, linestyle="--")
ax4.set_xlabel("Valores ajustados")
ax4.set_ylabel("Residuos")
ax4.set_title("Q2(c): Residuos vs Ajustados (Y~X, sin Chile)")
save_plot(fig4, filename="q2_c_resid_vs_fitted_no_chile")

fig5 = plt.figure(figsize=(6, 4))
qqplot(model_c.resid, line="s", ax=plt.gca())
plt.title("Q2(c): QQ-plot de residuos (Y~X, sin Chile)")
save_plot(plt.gcf(), filename="q2_c_resid_qqplot_no_chile")

exog_nc = sm.add_constant(T119_no_chile[["X"]])
BP_stat_c, BP_pval_c, fval_c, fpval_c = het_breuschpagan(model_c.resid, exog_nc)
W_stat_c, W_pval_c, _, _ = het_white(model_c.resid, exog_nc)

het_table_c = pd.DataFrame(
    {
        "Prueba": ["Breusch-Pagan", "White"],
        "Estadístico": [BP_stat_c, W_stat_c],
        "p-valor": [BP_pval_c, W_pval_c],
    }
)

save_latex_table(
    het_table_c,
    filename="q2_c_hetero_tests_no_chile.tex",
    rename_map={},
    caption="Pregunta 2(c): Pruebas de heterocedasticidad para Y~X (sin Chile)",
    label="tab:q2c_hetero",
)

# %%
comp = pd.DataFrame(
    {
        "Modelo": ["Todos", "Sin Chile"],
        "beta0": [
            model_b.params.get("Intercept", np.nan),
            model_c.params.get("Intercept", np.nan),
        ],
        "beta1": [model_b.params.get("X", np.nan), model_c.params.get("X", np.nan)],
        "R2": [model_b.rsquared, model_c.rsquared],
        "BP pval": [BP_pval, BP_pval_c],
        "White pval": [W_pval, W_pval_c],
    }
)

save_latex_table(
    comp,
    filename="q2_d_comparacion.tex",
    rename_map={
        "beta0": "$\\beta_0$",
        "beta1": "$\\beta_1$",
        "R2": "$R^2$",
        "BP pval": "BP p-valor",
        "White pval": "White p-valor",
    },
    caption="Pregunta 2(d): Comparación de OLS y pruebas de heterocedasticidad (todos vs. sin Chile)",
    label="tab:q2d_comp",
)

# %%
print("\nListo: Figuras guardadas en", PLOTS_DIR)
print("Tablas LaTeX guardadas en", LATEX_OUT)
