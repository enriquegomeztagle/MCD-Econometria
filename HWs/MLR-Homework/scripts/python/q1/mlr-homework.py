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
data = pd.DataFrame({"Y": [1, 3, 8], "X2": [1, 2, 3], "X3": [2, 1, -3]})
print(data)

# %%
import statsmodels.api as sm

# (1) Yi = α1 + α2*X2i + u
X1 = sm.add_constant(data["X2"])
model1 = sm.OLS(data["Y"], X1).fit()

# (2) Yi = λ1 + λ3*X3i + u
X2 = sm.add_constant(data["X3"])
model2 = sm.OLS(data["Y"], X2).fit()

# (3) Yi = β1 + β2*X2i + β3*X3i + u
X3 = sm.add_constant(data[["X2", "X3"]])
model3 = sm.OLS(data["Y"], X3).fit()

print("\nModelo 1:\n", model1.params)
print("\nModelo 2:\n", model2.params)
print("\nModelo 3:\n", model3.params)

# %%
alpha2 = model1.params["X2"]
beta2 = model3.params["X2"]

lambda3 = model2.params["X3"]
beta3 = model3.params["X3"]

print("α2 =", alpha2)
print("β2 =", beta2)
if abs(alpha2 - beta2) < 1e-6:
    print("Sí, α2 = β2")
else:
    print("No, α2 ≠ β2")

print("\nλ3 =", lambda3)
print("β3 =", beta3)
if abs(lambda3 - beta3) < 1e-6:
    print("Sí, λ3 = β3")
else:
    print("No, λ3 ≠ β3")

# %%
