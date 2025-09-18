# %%
import pandas as pd
import statsmodels.api as sm
from scipy.stats import jarque_bera
from statsmodels.stats.stattools import jarque_bera as jb_sm
import numpy as np

# %%
# Crear dataset dummy con 300 observaciones y 2 columnas
rng = np.random.default_rng(42)
datos = pd.DataFrame(
    {
        "variable_independiente": rng.standard_normal(300),
        "variable_dependiente": 5
        + 2 * rng.standard_normal(300)
        + rng.standard_normal(300),
    }
)

# %%
X = datos[["variable_independiente"]]
y = datos["variable_dependiente"]
X = sm.add_constant(X)  # Agrega el intercepto.
modelo = sm.OLS(y, X).fit()
residuos = modelo.resid

# %%
# JarqueBera con SciPy
jb_stat, jb_pvalue = jarque_bera(residuos)
print(f"JB: {jb_stat:.4f}")
print(f"p-valor: {jb_pvalue:.6f}")
alpha = 0.05

# %%
if jb_pvalue < alpha:
    print("Rechazamos H0: los residuos no son normales.")
else:
    print("No rechazamos H0: no hay evidencia contra la normalidad.")

# %%
jb_stat_sm, jb_pvalue_sm, skew, kurt = jb_sm(residuos)
print(
    f"(statsmodels) JB: {jb_stat_sm:.4f}, p-valor: {jb_pvalue_sm:.6f}, skew: {skew:.4f}, kurtosis: {kurt:.4f}"
)

# %%
