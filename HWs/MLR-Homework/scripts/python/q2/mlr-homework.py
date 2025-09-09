# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
data = pd.DataFrame(
    {
        "Y": [
            11484,
            9348,
            8429,
            10079,
            9240,
            8862,
            6216,
            8253,
            8038,
            7476,
            5911,
            7950,
            6134,
            5868,
            3160,
            5872,
        ],
        "X2": [
            2.26,
            2.54,
            3.07,
            2.91,
            2.73,
            2.77,
            3.59,
            3.23,
            2.60,
            2.89,
            3.77,
            3.64,
            2.82,
            2.96,
            4.24,
            3.69,
        ],
        "X3": [
            3.49,
            2.85,
            4.06,
            3.64,
            3.21,
            3.66,
            3.76,
            3.49,
            3.13,
            3.20,
            3.65,
            3.60,
            2.94,
            3.12,
            3.58,
            3.53,
        ],
        "X4": [
            158.11,
            173.36,
            165.26,
            172.92,
            178.46,
            198.62,
            186.28,
            188.98,
            180.49,
            183.33,
            181.87,
            185.00,
            184.00,
            188.20,
            175.67,
            188.00,
        ],
        "X5": list(range(1, 17)),
    }
)
data.head(20)

# %%
X_lin = sm.add_constant(data[["X2", "X3", "X4", "X5"]])
mod_lin = sm.OLS(data["Y"], X_lin).fit()
print(mod_lin.summary())

# %%
df_log = pd.DataFrame(
    {
        "const": 1.0,
        "lnX2": np.log(data["X2"]),
        "lnX3": np.log(data["X3"]),
        "lnX4": np.log(data["X4"]),
        "X5": data["X5"],
    }
)
y_log = np.log(data["Y"])
mod_log = sm.OLS(y_log, df_log).fit()
print(mod_log.summary())

# %%
signos_esperados = {"lnX2": "neg", "lnX3": "pos", "lnX4": "pos"}
b2, b3, b4 = mod_log.params["lnX2"], mod_log.params["lnX3"], mod_log.params["lnX4"]

print("\n--- (c) Signos esperados vs. resultados (modelo log-lineal) ---")
print(
    f"Elasticidad precio propio (b2) = {b2:.4f}  -> esperado: negativo | cumple? {b2 < 0}"
)
print(
    f"Elasticidad precio cruzado (b3) = {b3:.4f} -> esperado: positivo | cumple? {b3 > 0}"
)
print(
    f"Elasticidad ingreso (b4) = {b4:.4f}       -> esperado: positivo | cumple? {b4 > 0}"
)

# %%
a2, a3, a4 = mod_lin.params["X2"], mod_lin.params["X3"], mod_lin.params["X4"]
medias = data.mean()
elas_precio_propio_lin = a2 * (medias["X2"] / medias["Y"])
elas_precio_cruzado_lin = a3 * (medias["X3"] / medias["Y"])
elas_ingreso_lin = a4 * (medias["X4"] / medias["Y"])

print("\n--- (d) Elasticidades en el modelo lineal (evaluadas en medias) ---")
print(f"ε_precio_propio (lineal)  = {elas_precio_propio_lin:.4f}")
print(f"ε_precio_cruzado (lineal) = {elas_precio_cruzado_lin:.4f}")
print(f"ε_ingreso (lineal)        = {elas_ingreso_lin:.4f}")

print("\nElasticidades (log-lineal):")
print(f"ε_precio_propio (log)  = {b2:.4f}")
print(f"ε_precio_cruzado (log) = {b3:.4f}")
print(f"ε_ingreso (log)        = {b4:.4f}")

# %%
R2adj_lin, R2adj_log = mod_lin.rsquared_adj, mod_log.rsquared_adj
AIC_lin, AIC_log = mod_lin.aic, mod_log.aic
BIC_lin, BIC_log = mod_lin.bic, mod_log.bic

print("\n--- (e) Comparación de modelos ---")
print(f"R2 ajustado: lineal = {R2adj_lin:.4f}, log-lineal = {R2adj_log:.4f}")
print(f"AIC:          lineal = {AIC_lin:.2f},  log-lineal = {AIC_log:.2f}")
print(f"BIC:          lineal = {BIC_lin:.2f},  log-lineal = {BIC_log:.2f}")

if R2adj_lin > R2adj_log:
    mejor_R2 = "lineal"
else:
    mejor_R2 = "log-lineal"

votos = {"lineal": 0, "log-lineal": 0}
votos["lineal"] += (AIC_lin < AIC_log) + (BIC_lin < BIC_log)
votos["log-lineal"] += (AIC_log < AIC_lin) + (BIC_log < BIC_lin)
mejor_ic = max(votos, key=votos.get)

print(f"\nMejor por R2 ajustado → {mejor_R2}")
print(f"Mejor por (AIC,BIC)   → {mejor_ic}  (votos: {votos})")

if mejor_ic == "log-lineal":
    recomendacion = "log-lineal"
else:
    recomendacion = "lineal"

print(f"\nRecomendación final (por IC): usar el modelo {recomendacion}.")
print(
    "Nota: si tu objetivo principal es reportar ELASTICIDADES, prefiere el log-lineal;"
)
print("si priorizas ajuste en niveles para predicción, el lineal puede ser preferido.")

# %%
