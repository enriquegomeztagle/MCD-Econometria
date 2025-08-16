# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv("../../data/datos_fuerza_levantamiento.csv")
df.head()

# %%
print("Generando gráfico de dispersión...")
plt.figure(figsize=(6, 4))
plt.scatter(df["Fuerza_del_brazo_x"], df["Levantamiento_dinamico_y"])
plt.xlabel("Fuerza del brazo, x")
plt.ylabel("Levantamiento dinámico, y")
plt.title("Dispersión: y vs. x")
plt.grid(True, linestyle="--", linewidth=0.5)
import os

output_dir = "../../plots/python/ejercicio1"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "scatter_plot.png"), dpi=300, bbox_inches="tight")
plt.show()

# %%
print("Generando estadísticas descriptivas...")
desc = df[["Fuerza_del_brazo_x", "Levantamiento_dinamico_y"]].describe().T
print(desc)

# %%
print("Calculando coeficiente de correlación de Pearson...")
x = df["Fuerza_del_brazo_x"]
y = df["Levantamiento_dinamico_y"]

r, pval = stats.pearsonr(x, y)
print(f"r (Pearson) = {r:.4f}")
print(f"p-valor = {pval:.6f}")

print("Realizando prueba de hipótesis...")
alpha = 0.05
if pval < alpha:
    print("Conclusión: se rechaza H0; evidencia de relación lineal (α=0.05).")
else:
    print(
        "Conclusión: no se rechaza H0; no hay evidencia suficiente de relación lineal (α=0.05)."
    )

# %%
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print("Parámetros estimados:")
print(model.params)

print("\nResumen del modelo:")
print(model.summary())

# %%
print("Estimando modelo de regresión lineal simple por MCO...")
x0 = 30.0

exog_names = model.model.exog_names
X_new = pd.DataFrame([[1.0, x0]], columns=exog_names)

pred = model.get_prediction(X_new)
pred_summary = pred.summary_frame(alpha=0.05)

print(f"Estimación puntual E[y|x={x0}] = {pred_summary.loc[0, 'mean']:.4f}")
print("Intervalo de confianza al 95% para la media condicional:")
print(pred_summary[["mean_ci_lower", "mean_ci_upper"]])

print("\nIntervalo de predicción al 95% (para una observación nueva):")
print(pred_summary[["obs_ci_lower", "obs_ci_upper"]])

# %%
resid = model.resid
print("Generando gráfico de residuales...")
plt.figure(figsize=(6, 4))
plt.scatter(x, resid)
plt.axhline(0, linestyle="--")
plt.xlabel("Fuerza del brazo, x")
plt.ylabel("Residuales")
plt.title("Residuales vs. x")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.savefig(
    "../../plots/python/ejercicio1/residuals_plot.png", dpi=300, bbox_inches="tight"
)
plt.show()

print(f"Media de residuales (debe ser ~0): {resid.mean():.6f}")
print(f"Desviación estándar de residuales: {resid.std(ddof=1):.6f}")
corr_rx = np.corrcoef(x, resid)[0, 1]
print(
    f"Correlación(x, residuales) = {corr_rx:.6f} (debe ser cercana a 0 en MCO con intercepto)"
)

# %%
