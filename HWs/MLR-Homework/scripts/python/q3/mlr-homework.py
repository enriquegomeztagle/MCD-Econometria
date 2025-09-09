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
DATA_DIR = Path("../../../data/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LATEX_OUT = Path("../../../docs/latex_utils/tables")
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
years = list(range(1962, 1982))
Y = [
    51.1,
    52.3,
    53.6,
    49.6,
    56.8,
    70.1,
    80.5,
    81.2,
    80.3,
    77.7,
    78.3,
    74.5,
    77.8,
    85.6,
    89.4,
    97.5,
    105.2,
    117.7,
    135.9,
    162.1,
]
X2 = [
    560.3,
    590.5,
    632.4,
    684.9,
    749.9,
    793.0,
    865.0,
    931.4,
    992.7,
    1077.6,
    1185.9,
    1326.4,
    1434.2,
    1549.2,
    1748.0,
    1918.3,
    2163.9,
    2417.8,
    2633.1,
    2937.7,
]
X3 = [
    0.6,
    0.9,
    1.1,
    1.4,
    1.6,
    1.0,
    0.8,
    1.5,
    1.0,
    1.5,
    2.95,
    4.8,
    10.3,
    16.0,
    14.7,
    8.3,
    11.0,
    13.0,
    15.3,
    18.0,
]
X4 = [
    16.0,
    16.4,
    16.7,
    17.0,
    20.2,
    23.1,
    25.6,
    24.6,
    24.8,
    27.1,
    21.5,
    24.3,
    26.8,
    29.5,
    30.4,
    33.3,
    38.0,
    46.2,
    57.6,
    68.9,
]
X5 = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

df = pd.DataFrame({"year": years, "Y": Y, "X2": X2, "X3": X3, "X4": X4, "X5": X5})
df.head(30)

# %%
X = sm.add_constant(df[["X2", "X3", "X4", "X5"]])
ols = sm.OLS(df["Y"], X).fit()
print(ols.summary())

# %%
b = ols.params
se = ols.bse
print("\n--- Coeficientes centrales ---")
for name in ["const", "X2", "X3", "X4", "X5"]:
    print(f"{name:>6}: {b[name]:10.4f} (se = {se[name]:.4f})")

print(f"\nR2 = {ols.rsquared:.4f} | R2 ajustado = {ols.rsquared_adj:.4f}")

# %%
print("\n--- (b) Signos esperados vs estimados ---")
esperados = {"X2": "pos", "X3": "pos", "X4": "pos", "X5": "pos"}


def cumple_signo(valor, esperado):
    return (valor > 0) if esperado == "pos" else (valor < 0)


for var, sgn in esperados.items():
    ok = cumple_signo(b[var], sgn)
    print(f"{var}: estimado = {b[var]:.4f} | esperado {sgn} -> ¿cumple? {ok}")

efecto_dummy = b["X5"]
print(
    f"\nLa dummy X5 (conflictos ≥100k) aumenta Y en {efecto_dummy:.2f} miles de millones (ceteris paribus)."
)

# %%
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_no_const = df[["X2", "X3", "X4", "X5"]].astype(float).values
    vif = [variance_inflation_factor(X_no_const, i) for i in range(X_no_const.shape[1])]
    print("\n--- VIF (multicolinealidad; >10 es alto) ---")
    for name, v in zip(["X2", "X3", "X4", "X5"], vif):
        print(f"{name}: VIF = {v:.2f}")
except Exception as e:
    print("No se pudo calcular VIF:", e)


# %%
def signo(x):
    return "+" if x > 0 else "-"


txt = (
    f"\nResumen rápido:\n"
    f" • b2 (PNB)  = {b['X2']:.4f} ({signo(b['X2'])}) — esperado positivo.\n"
    f" • b3 (Mil)  = {b['X3']:.4f} ({signo(b['X3'])}) — esperado positivo.\n"
    f" • b4 (Aero) = {b['X4']:.4f} ({signo(b['X4'])}) — esperado positivo.\n"
    f" • b5 (Dummy conflictos) = {b['X5']:.4f} ({signo(b['X5'])}).\n"
    f"R2_aj = {ols.rsquared_adj:.3f}. "
    f"Conclusión: signos y magnitudes comparables con expectativas a priori "
    f"(crecimiento económico, actividad militar/aeroespacial y guerras elevan el presupuesto)."
)
print(txt)

# %%
