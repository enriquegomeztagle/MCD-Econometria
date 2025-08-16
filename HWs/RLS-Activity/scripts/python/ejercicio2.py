# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "../../data/shampoo.csv"
PLOTS_DIR = "../../plots/python/ejercicio2"
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Cargando:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("\nPrimeras filas:")
print(df.head())

ventas_col_candidates = ["Ventas_Millones_lts", "Ventas", "ventas"]
inversion_col_candidates = ["Inversion_Millones_pesos", "Inversion", "inversion"]

ventas_col = next(c for c in ventas_col_candidates if c in df.columns)
inversion_col = next(c for c in inversion_col_candidates if c in df.columns)

x = df[inversion_col].astype(float)
y = df[ventas_col].astype(float)

n = len(df)
print(f"\nColumnas detectadas -> y: {ventas_col}, x: {inversion_col}; n = {n}")
# %%
print("\n(a) Generando diagrama de dispersión y estadísticas descriptivas...")
plt.figure(figsize=(6.5, 4.5))
plt.scatter(x, y)
plt.xlabel("Inversión en redes (millones de pesos)")
plt.ylabel("Ventas (millones de litros)")
plt.title("Dispersión: Ventas vs. Inversión")
plt.grid(True, linestyle="--", linewidth=0.5)
scatter_path = os.path.join(PLOTS_DIR, "scatter_plot.png")
plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
print("Figura guardada en:", scatter_path)
plt.show()

desc = pd.DataFrame(
    {
        "media": [y.mean(), x.mean()],
        "mediana": [y.median(), x.median()],
        "desv_est": [y.std(ddof=1), x.std(ddof=1)],
        "min": [y.min(), x.min()],
        "Q1": [y.quantile(0.25), x.quantile(0.25)],
        "Q3": [y.quantile(0.75), x.quantile(0.75)],
        "max": [y.max(), x.max()],
    },
    index=["Ventas (y)", "Inversión (x)"],
)
print("\nEstadísticas descriptivas:")
print(desc.round(3))

# %%
print("\n(b) Correlación de Pearson e intervalo de confianza al 95%...")
r, pval = stats.pearsonr(x, y)
print(f"r = {r:.4f}")
print(f"p-valor (bilateral) = {pval:.6f}")

# IC de r vía transformación z de Fisher
z = np.arctanh(r)  # Fisher z
se_z = 1 / np.sqrt(n - 3)
z_crit = stats.norm.ppf(0.975)  # 1.96
lo_z, hi_z = z - z_crit * se_z, z + z_crit * se_z
lo_r, hi_r = np.tanh(lo_z), np.tanh(hi_z)
print(f"IC 95% para r: [{lo_r:.4f}, {hi_r:.4f}]")

# %%
print("\n(c) Ajustando RLS (MCO): y = beta0 + beta1*x + e ...")
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print("\nParámetros estimados:")
print(model.params)
print("\nResumen del modelo:")
print(model.summary())

# %%
print("\n(d) Prueba de hipótesis (one-sided, alpha=0.05)...")

beta1_hat = model.params[1]
se_beta1 = model.bse[1]
df_resid = int(model.df_resid)

beta1_H0 = 0.1

t_stat = (beta1_hat - beta1_H0) / se_beta1
p_one_sided = 1 - stats.t.cdf(t_stat, df=df_resid)  # H1: beta1 > 0.1

print(f"beta1_hat = {beta1_hat:.6f}")
print(f"SE(beta1) = {se_beta1:.6f}")
print(f"t = {t_stat:.4f}, df = {df_resid}")
print(f"p-valor (H1: beta1 > 0.1) = {p_one_sided:.6f}")

alpha = 0.05
if p_one_sided < alpha:
    print(
        "Conclusión: Se RECHAZA H0. La evidencia sugiere un incremento > 50 mil litros por cada 500 mil pesos."
    )
else:
    print(
        "Conclusión: NO se rechaza H0 al 5%. No hay evidencia suficiente para afirmar un incremento > 50 mil litros por cada 500 mil pesos."
    )

# %%
plt.figure(figsize=(6.5, 4.5))
plt.scatter(x, y, label="Datos")
xx = np.linspace(x.min(), x.max(), 100)
X_line = sm.add_constant(xx)
y_hat = model.predict(X_line)
plt.plot(xx, y_hat, label="Recta ajustada")
plt.xlabel("Inversión en redes (millones de pesos)")
plt.ylabel("Ventas (millones de litros)")
plt.title("RLS: Ventas ~ Inversión")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
line_path = os.path.join(PLOTS_DIR, "shampoo_rls_line.png")
plt.savefig(line_path, dpi=300, bbox_inches="tight")
print("Figura guardada en:", line_path)
plt.show()
