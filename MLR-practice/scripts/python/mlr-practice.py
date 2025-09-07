# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import os
from pathlib import Path

# %%
DATA_DIR = Path("../../data/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = Path("../../plots/python/")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TAB_OUT = Path("../../data/processed")
TAB_OUT.mkdir(parents=True, exist_ok=True)

LATEX_OUT = Path("../../docs/latex_utils/tables")
LATEX_OUT.mkdir(parents=True, exist_ok=True)


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
df = pd.read_excel(os.path.join(DATA_DIR, "DatosPracticaRLM.xlsx"), header=1)
df = df.rename(columns={"educación": "educacion", "Unnamed: 0": "estado"})
print(df.head())

rename_map = {
    "estado": "Estado",
    "educación": "Educación",
    "ingreso": "Ingreso",
    "menores": "Menores",
    "urbano": "Urbano",
}

save_latex_table(
    df.head(),
    filename="tabla_head_datos.tex",
    rename_map=rename_map,
    caption="Primeras filas del conjunto de datos utilizado en el análisis",
    label="tab:head_datos",
)


# %%
def get_avg_per_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    print("Promedio de la columna", column_name, "es:", df[column_name].mean())


get_avg_per_column(df, "ingreso")
get_avg_per_column(df, "menores")
get_avg_per_column(df, "urbano")

# %%
model = smf.ols("educacion ~ ingreso + menores + urbano", data=df).fit()
print(model.summary())

# %%
coef_table = pd.DataFrame(
    {
        "Coeficiente": model.params,
        "Error Std.": model.bse,
        "t": model.tvalues,
        "p-valor": model.pvalues,
    }
)
coef_table["IC 2.5%"] = model.conf_int()[0]
coef_table["IC 97.5%"] = model.conf_int()[1]
coef_table.index.name = "Parámetro"
print("\nTabla de coeficientes:\n", coef_table)

rename_map = {
    "Coeficiente": "Coef.",
    "Error Std.": "Err. Std.",
    "p-valor": "p",
    "IC 2.5%": "IC 2.5\\%",
    "IC 97.5%": "IC 97.5\\%",
}

coef_table_out = coef_table.reset_index().rename(columns={"Parámetro": "Parámetro"})
save_latex_table(
    df=coef_table_out.rename(columns=rename_map),
    filename="reg_multiple_coeficientes.tex",
    rename_map={},
    caption="Coeficientes del modelo OLS: Educación vs Ingreso, Menores y Urbano",
    label="tab:ols_coef_educacion",
)

# %%
df["y_hat"] = model.fittedvalues
df["resid"] = model.resid
print(df[["educacion", "y_hat", "resid"]].head())

# %%
metrics = {
    "R²": model.rsquared,
    "R² ajustado": model.rsquared_adj,
    "Estadístico F": model.fvalue,
    "p-valor (F)": model.f_pvalue,
    "AIC": model.aic,
    "BIC": model.bic,
    "Observaciones": model.nobs,
}

metrics_df = pd.DataFrame(list(metrics.items()), columns=["Métrica", "Valor"])
print(metrics_df)

save_latex_table(
    df=metrics_df,
    filename="tabla_metricas_regresion.tex",
    rename_map={"Métrica": "Métrica", "Valor": "Valor"},
    caption="Métricas globales del modelo de regresión lineal múltiple",
    label="tab:metricas_regresion",
)

# %%
S2 = model.ssr / model.df_resid
cov_matrix = model.cov_params()
std_errors = model.bse
print("Estimación de la varianza del error (S²):", S2)
print("\nErrores estándar de los coeficientes:")
print(std_errors)

S2_df = pd.DataFrame({"Métrica": ["S^2 (Varianza del error)"], "Valor": [S2]})
save_latex_table(
    df=S2_df,
    filename="varianza_error.tex",
    rename_map={"Métrica": "Métrica", "Valor": "Valor"},
    caption="Estimación de la varianza del error $S^2$ del modelo OLS.",
    label="tab:varianza_error",
)

coef_df = pd.DataFrame(
    {
        "Coeficiente": model.params,
        "Error Estándar": model.bse,
        "t-valor": model.tvalues,
        "p-valor": model.pvalues,
    }
)

save_latex_table(
    df=coef_df,
    filename="tabla_coeficientes.tex",
    rename_map={
        "Coeficiente": "Coeficiente",
        "Error Estándar": "Error Estándar",
        "t-valor": "t",
        "p-valor": "p",
    },
    caption="Estimaciones, errores estándar y pruebas t de los coeficientes del modelo",
    label="tab:coeficientes_regresion",
)

cov_df = cov_matrix.copy()
cov_df.index.name = "Parámetro"
cov_df = cov_df.reset_index()

save_latex_table(
    df=cov_df,
    filename="matriz_covarianzas_parametros.tex",
    rename_map={},
    caption="Matriz de covarianzas de los estimadores del modelo OLS.",
    label="tab:covarianzas_parametros",
)

# %%
conf_int = 0.9
alpha = 1 - conf_int

conf_int_90 = model.conf_int(alpha=alpha)
conf_int_90.columns = ["Límite inferior", "Límite superior"]

beta2_ci = conf_int_90.loc[["ingreso"]]
print("Intervalo de confianza al 90% para β2 (Ingreso):")
print(
    f"[{beta2_ci['Límite inferior'].values[0]:.4f}, {beta2_ci['Límite superior'].values[0]:.4f}]"
)

save_latex_table(
    df=beta2_ci.reset_index().rename(columns={"index": "Parámetro"}),
    filename="intervalo_confianza_beta2.tex",
    rename_map={
        "index": "Parámetro",
        "Límite inferior": "Límite inferior",
        "Límite superior": "Límite superior",
    },
    caption="Intervalo de confianza del 90\% para el coeficiente $\\beta_2$ (Ingreso).",
    label="tab:ci_beta2",
)

# %%
x_means = df[["ingreso", "menores", "urbano"]].mean()

y_hat_mean = (
    model.params["Intercept"]
    + model.params["ingreso"] * x_means["ingreso"]
    + model.params["menores"] * x_means["menores"]
    + model.params["urbano"] * x_means["urbano"]
)

mean_point_df = pd.DataFrame(
    {
        "Variable": ["Ingreso", "Menores", "Urbano", "Educación esperada"],
        "Valor promedio": [
            round(x_means["ingreso"], 2),
            round(x_means["menores"], 2),
            round(x_means["urbano"], 2),
            round(y_hat_mean, 2),
        ],
    }
)

print(mean_point_df)

save_latex_table(
    df=mean_point_df,
    filename="tabla_prediccion_media.tex",
    rename_map={"Variable": "Variable", "Valor promedio": "Valor promedio"},
    caption="Valores promedio de los regresores y gasto en educación esperado en el punto promedio",
    label="tab:prediccion_media",
)

# %%
n = int(model.nobs)
k = int(model.df_model)
df1 = k
df2 = int(model.df_resid)

F_stat = model.fvalue
p_value_F = model.f_pvalue

alpha = 0.05
F_crit = stats.f.ppf(1 - alpha, df1, df2)

print("Hipótesis:")
print("H0: β1 = β2 = β3 = 0 (el modelo no tiene poder explicativo)")
print("H1: Al menos un βj ≠ 0")

print("\nEstadístico F:", F_stat)
print("Grados de libertad:", df1, "y", df2)
print("Valor crítico F (α=0.05):", F_crit)
print("p-valor (F):", p_value_F)

if F_stat > F_crit:
    conclusion = "Se RECHAZA H0: el modelo es globalmente significativo."
else:
    conclusion = "No se rechaza H0: el modelo no es significativo."

print("\nConclusión:", conclusion)

f_test_df = pd.DataFrame(
    {
        "Estadístico": ["F observado", "F crítico (α=0.05)", "p-valor", "gl1", "gl2"],
        "Valor": [round(F_stat, 3), round(F_crit, 3), f"{p_value_F:.3g}", df1, df2],
    }
)

save_latex_table(
    df=f_test_df,
    filename="prueba_F_modelo.tex",
    rename_map={"Estadístico": "Estadístico", "Valor": "Valor"},
    caption="Prueba de significancia global del modelo de regresión (estadístico F).",
    label="tab:prueba_f",
)

# %%
hipotesis_df = pd.DataFrame(
    {
        "Elemento": ["Hipótesis nula (H0)", "Hipótesis alternativa (H1)", "Conclusión"],
        "Expresión": [
            r"$\beta_1 = \beta_2 = \beta_3 = 0$ (el modelo no es significativo)",
            r"Al menos un $\beta_j \neq 0$",
            conclusion,
        ],
    }
)

print(hipotesis_df)

save_latex_table(
    df=hipotesis_df,
    filename="hipotesis_y_conclusion.tex",
    rename_map={"Elemento": "Elemento", "Expresión": "Descripción"},
    caption="Hipótesis y conclusión de la prueba F global del modelo de regresión",
    label="tab:hipotesis_conclusion",
)

# %%
