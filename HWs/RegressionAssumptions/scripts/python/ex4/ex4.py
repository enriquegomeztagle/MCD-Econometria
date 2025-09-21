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

LATEX_OUT = Path("../../docs/latex_utils/tables")
LATEX_OUT.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = Path("../../plots/python/ex2/")
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
df = pd.read_excel(os.path.join(DATA_DIR, "Table11_9.xlsx"), header=1)
print(df.head())
