# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

DATA_CANDIDATES = [
    "../../data/flowers.xlsx",
    "../../data/flowers.csv",
]
PLOTS_DIR = "../../plots/python/ejercicio3"
os.makedirs(PLOTS_DIR, exist_ok=True)

for path in DATA_CANDIDATES:
    if os.path.exists(path):
        DATA_PATH = path
        break
else:
    raise FileNotFoundError(
        "No se encontró ningún archivo de datos. Esperaba ../data/flowers.xlsx o ../data/flowers.csv"
    )

print("Cargando:", DATA_PATH)
if DATA_PATH.endswith(".xlsx"):
    df = pd.read_excel(DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH)

print("\nPrimeras filas:")
print(df.head())

x_candidates = ["flores", "Flores", "X", "x"]
y_candidates = ["producción", "produccion", "Producción", "Y", "y"]

x_col = next(c for c in x_candidates if c in df.columns)
y_col = next(c for c in y_candidates if c in df.columns)

x = df[x_col].astype(float)
y = df[y_col].astype(float)

n = len(df)
print(f"\nColumnas detectadas -> x: {x_col}, y: {y_col}; n = {n}")

# %%
print("\n(a) Gráfico de dispersión y estadísticas descriptivas...")
plt.figure(figsize=(6.8, 4.8))
plt.scatter(x, y)
plt.xlabel("Flores procesadas, x (miles)")
plt.ylabel("Producción de esencia, y (onzas)")
plt.title("Dispersión: Producción vs. Flores")
plt.grid(True, linestyle='--', linewidth=0.5)
scatter_path = os.path.join(PLOTS_DIR, "scatter_flowers.png")
plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
print("Figura guardada en:", scatter_path)
plt.show()

desc = pd.DataFrame({
    "media": [x.mean(), y.mean()],
    "mediana": [x.median(), y.median()],
    "desv_est": [x.std(ddof=1), y.std(ddof=1)],
    "min": [x.min(), y.min()],
    "Q1": [x.quantile(0.25), y.quantile(0.25)],
    "Q3": [x.quantile(0.75), y.quantile(0.75)],
    "max": [x.max(), y.max()],
}, index=["x (flores)", "y (producción)"])
print("\nEstadísticas descriptivas (redondeadas):")
print(desc.round(3))

# %%
print("\n(b) Relación lineal: signo y evidencia estadística...")
r, pval = stats.pearsonr(x, y)
print(f"Coef. de correlación de Pearson r = {r:.4f}")
print(f"p-valor (bilateral) = {pval:.6f}")
if r > 0:
    print("Dirección: positiva")
elif r < 0:
    print("Dirección: negativa")
else:
    print("Dirección: nula")

z = np.arctanh(r); se_z = 1/np.sqrt(n-3); z_crit = stats.norm.ppf(0.975)
lo_r, hi_r = np.tanh(z - z_crit*se_z), np.tanh(z + z_crit*se_z)
print(f"IC 95% de r: [{lo_r:.4f}, {hi_r:.4f}]")

# %%
print("\n(c) Ajuste RLS (MCO) y verificación de b0, b1, S^2...")
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
params = model.params
b0_hat = params[0]
b1_hat = params[1]

resid = model.resid
SSE = np.sum(resid**2)
S2 = SSE/(n-2)

print(f"b0_hat = {b0_hat:.4f}")
print(f"b1_hat = {b1_hat:.4f}")
print(f"S^2 (SSE/(n-2)) = {S2:.4f}")
print("\nResumen del modelo:")
print(model.summary())

plt.figure(figsize=(6.8, 4.8))
plt.scatter(x, y, label="Datos")
xx = np.linspace(x.min(), x.max(), 100)
X_line = sm.add_constant(xx)
y_hat = model.predict(X_line)
plt.plot(xx, y_hat, label="Recta ajustada")
plt.xlabel("Flores procesadas, x (miles)")
plt.ylabel("Producción de esencia, y (onzas)")
plt.title("RLS: y ~ x")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
line_path = os.path.join(PLOTS_DIR, "rls_line.png")
plt.savefig(line_path, dpi=300, bbox_inches="tight")
print("Figura guardada en:", line_path)
plt.show()

# %%
print("\n(c.1) Cálculos manuales para trazabilidad y verificación...")
Sxx = np.sum((x - x.mean())**2)
Syy = np.sum((y - y.mean())**2)
Sxy = np.sum((x - x.mean())*(y - y.mean()))

b1_manual = Sxy / Sxx
b0_manual = y.mean() - b1_manual * x.mean()
SSE_manual = np.sum((y - (b0_manual + b1_manual*x))**2)
S2_manual = SSE_manual / (n - 2)

print(f"Sxx = {Sxx:.6f}, Syy = {Syy:.6f}, Sxy = {Sxy:.6f}")
print(f"b0_manual = {b0_manual:.4f}, b1_manual = {b1_manual:.4f}")
print(f"S^2_manual = {S2_manual:.4f}")

b0_ref, b1_ref, S2_ref = 1.38, 0.52, 0.206
print("\nComparación con referencia (tolerancia +/- 0.03 para b0/b1 y +/- 0.02 para S^2):")
print(f"|b0_manual - {b0_ref}| = {abs(b0_manual - b0_ref):.4f}")
print(f"|b1_manual - {b1_ref}| = {abs(b1_manual - b1_ref):.4f}")
print(f"|S^2_manual - {S2_ref}| = {abs(S2_manual - S2_ref):.4f}")
print("Coincide b0? ", abs(b0_manual - b0_ref) <= 0.03)
print("Coincide b1? ", abs(b1_manual - b1_ref) <= 0.03)
print("Coincide S^2? ", abs(S2_manual - S2_ref) <= 0.02)

# %% 
print("\n(d) ANOVA de la regresión y prueba F...")
y_bar = y.mean()
SSR = np.sum((model.fittedvalues - y_bar)**2)
SSE = np.sum((y - model.fittedvalues)**2)
SST = SSR + SSE

DF_model = 1
DF_resid = n - 2
DF_total = n - 1

MSR = SSR/DF_model
MSE = SSE/DF_resid
F_stat = MSR/MSE
p_F = 1 - stats.f.cdf(F_stat, DF_model, DF_resid)

anova = pd.DataFrame({
    "SC": [SSR, SSE, SST],
    "gl": [DF_model, DF_resid, DF_total],
    "CM": [MSR, MSE, "-"],
}, index=["Regresión", "Error", "Total"])
print("Tabla ANOVA (redondeada):")
print(anova.round(4))
print(f"F = {F_stat:.4f}, df1 = {DF_model}, df2 = {DF_resid}, p-valor = {p_F:.6f}")

# %% 
print("\n(d.1) Diagnósticos del modelo...")
resid = model.resid
fitted = model.fittedvalues

plt.figure(figsize=(6.8, 4.6))
plt.scatter(fitted, resid)
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel("Valores ajustados")
plt.ylabel("Residuales")
plt.title("Residuales vs. Ajustados")
plt.grid(True, linestyle='--', linewidth=0.5)
resid_fit_path = os.path.join(PLOTS_DIR, "residuals_vs_fitted.png")
plt.savefig(resid_fit_path, dpi=300, bbox_inches="tight")
print("Figura guardada en:", resid_fit_path)
plt.show()

fig = sm.qqplot(resid, line='45')
plt.title("QQ-plot de residuales")
qq_path = os.path.join(PLOTS_DIR, "qqplot_residuals.png")
plt.savefig(qq_path, dpi=300, bbox_inches="tight")
print("Figura guardada en:", qq_path)
plt.show()

plt.figure(figsize=(6.8,4.6))
plt.hist(resid, bins=8, edgecolor='black')
plt.title("Histograma de residuales")
plt.xlabel("Residual")
plt.ylabel("Frecuencia")
hist_path = os.path.join(PLOTS_DIR, "hist_residuals.png")
plt.savefig(hist_path, dpi=300, bbox_inches="tight")
print("Figura guardada en:", hist_path)
plt.show()

jb_stat, jb_p, skew, kurt = jarque_bera(resid)
print(f"Jarque-Bera: JB = {jb_stat:.4f}, p = {jb_p:.6f}, skew = {skew:.4f}, kurt = {kurt:.4f}")

bp_stat, bp_p, fval, fp = het_breuschpagan(resid, sm.add_constant(fitted))
print(f"Breusch-Pagan: LM = {bp_stat:.4f}, p = {bp_p:.6f}; F = {fval:.4f}, p(F) = {fp:.6f}")

# %%
print("\n(e) Error estándar de la pendiente e IC al 95%...")
Sxx = np.sum((x - x.mean())**2)
se_b1 = np.sqrt(MSE / Sxx)
t_crit = stats.t.ppf(0.975, DF_resid)
ci_b1 = (b1_hat - t_crit*se_b1, b1_hat + t_crit*se_b1)
print(f"SE(b1) = {se_b1:.6f}")
print(f"IC 95% para b1: [{ci_b1[0]:.4f}, {ci_b1[1]:.4f}]")

# %%
print("\n(f) Porcentaje de variabilidad explicada...")
R2 = SSR / SST
print(f"R^2 = {R2:.4f} -> {100*R2:.2f}% de la variabilidad de y explicada por el modelo")

# %%
print("\n(g) IC 95% para la media condicional en x0 = 1.25...")
x0 = 1.25
exog_names = model.model.exog_names
X0 = pd.DataFrame([[1.0, x0]], columns=exog_names)
pred_mean = model.get_prediction(X0)
sum_mean = pred_mean.summary_frame(alpha=0.05)
print(sum_mean[["mean", "mean_ci_lower", "mean_ci_upper"]].round(4))

se_mean = np.sqrt(MSE * (1/n + (x0 - x.mean())**2 / Sxx))
mean_hat = float((X0.values @ model.params.values).ravel())
ci_mean = (mean_hat - t_crit*se_mean, mean_hat + t_crit*se_mean)
print(f"(Chequeo) E[y|x0] puntual = {mean_hat:.4f}")
print(f"(Chequeo) IC 95% manual para E[y|x0]: [{ci_mean[0]:.4f}, {ci_mean[1]:.4f}]")

# %%
print("\n(h) Intervalo de predicción al 95% en x0 = 1.95...")
x0 = 1.95
exog_names = model.model.exog_names
X0 = pd.DataFrame([[1.0, x0]], columns=exog_names)
pred_obs = model.get_prediction(X0)
sum_obs = pred_obs.summary_frame(alpha=0.05)
print(sum_obs[["mean", "obs_ci_lower", "obs_ci_upper"]].round(4))

se_pred = np.sqrt(MSE * (1 + 1/n + (x0 - x.mean())**2 / Sxx))
mean_hat = float((X0.values @ model.params.values).ravel())
pi = (mean_hat - t_crit*se_pred, mean_hat + t_crit*se_pred)
print(f"(Chequeo) y_hat puntual = {mean_hat:.4f}")
print(f"(Chequeo) PI 95% manual para y nueva: [{pi[0]:.4f}, {pi[1]:.4f}]")
