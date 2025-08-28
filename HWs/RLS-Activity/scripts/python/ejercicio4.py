# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

DATA_CANDIDATES = [
    "../../data/cableTV.xlsx",
    "../../data/cableTV.csv",
]
PLOTS_DIR = "../../plots/python/ejercicio4"
os.makedirs(PLOTS_DIR, exist_ok=True)

for path in DATA_CANDIDATES:
    if os.path.exists(path):
        DATA_PATH = path
        break
else:
    raise FileNotFoundError("No se encontró cableTV.{xlsx,csv} en ../data/")

print("Cargando:", DATA_PATH)
if DATA_PATH.endswith(".xlsx"):
    df = pd.read_excel(DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH)

print("\nColumnas disponibles:", list(df.columns))

rename_map = {
    "obs": "obs",
    "colonia": "colonia",
    "manzana": "manzana",
    "adultos": "adultos",
    "ninos": "ninos",
    "teles": "teles",
    "renta": "renta",
    "tvtot": "tvtot",
    "tipo": "tipo",
    "valor": "valor",
}
df.columns = [c.strip().lower() for c in df.columns]

expected = set(rename_map.keys())
missing = expected - set(df.columns)
if missing:
    print("Advertencia: faltan columnas esperadas:", missing)

df["x_valor_miles"] = df["valor"] / 1000.0
x_name = "x_valor_miles"
y_name = "renta"
print("\nPrimeras filas:")
print(
    df[
        [
            "obs",
            "colonia",
            "manzana",
            "adultos",
            "ninos",
            "teles",
            y_name,
            "tvtot",
            "tipo",
            "valor",
            x_name,
        ]
    ].head()
)


def fit_ols_xy(x, y):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model


# %%
print("\n(a) Ajuste MCO con todos los datos y gráficas...")
mask_full = np.ones(len(df), dtype=bool)
x = df.loc[mask_full, x_name].astype(float)
y = df.loc[mask_full, y_name].astype(float)
model_full = fit_ols_xy(x, y)

print("\nParámetros (todos los datos):")
print(model_full.params)

MSE_full = np.sum(model_full.resid**2) / model_full.df_resid
sigma_full = np.sqrt(MSE_full)
print(f"Sigma (EE de la regresión) = {sigma_full:.6f}")

plt.figure(figsize=(7.2, 5.0))
plt.scatter(x, y, label="Datos")
xx = np.linspace(x.min(), x.max(), 200)
X_line = sm.add_constant(xx)
y_hat = model_full.predict(X_line)
plt.plot(xx, y_hat, label="Recta ajustada")
plt.xlabel("Valor catastral (miles de pesos)")
plt.ylabel("Renta mensual (múltiplos de $5)")
plt.title("RLS (todos): Renta ~ Valor")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plot_path = os.path.join(PLOTS_DIR, "full_scatter_line.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print("Figura guardada en:", plot_path)
plt.show()

plt.figure(figsize=(7.2, 5.0))
plt.scatter(x, model_full.resid)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Valor catastral (miles de pesos)")
plt.ylabel("Residuales")
plt.title("Residuales vs x (todos)")
plt.grid(True, linestyle="--", linewidth=0.5)
resid_path = os.path.join(PLOTS_DIR, "full_resid_vs_x.png")
plt.savefig(resid_path, dpi=300, bbox_inches="tight")
print("Figura guardada en:", resid_path)
plt.show()

# %%
print("\n(b) ANOVA y significancia — todos los datos")
X_full = sm.add_constant(x)
model_full = sm.OLS(y, X_full).fit()

SS_total = np.sum((y - np.mean(y)) ** 2)
SS_model = model_full.ess
SS_resid = model_full.ssr
df_model = model_full.df_model
df_resid = model_full.df_resid
df_total = df_model + df_resid

MS_model = SS_model / df_model
MS_resid = SS_resid / df_resid
F_stat = MS_model / MS_resid
p_value = 1 - stats.f.cdf(F_stat, df_model, df_resid)

anova_data = {
    "df": [df_model, df_resid, df_total],
    "sum_sq": [SS_model, SS_resid, SS_total],
    "mean_sq": [MS_model, MS_resid, np.nan],
    "F": [F_stat, np.nan, np.nan],
    "PR(>F)": [p_value, np.nan, np.nan],
}
anova_full = pd.DataFrame(anova_data, index=["x_valor_miles", "Residual", "Total"])
print("\nANOVA (todos):\n", anova_full.round(6))

F_full = F_stat
p_full = p_value
R2_full = model_full.rsquared
print(f"F = {F_full:.6f}, p-valor = {p_full:.6f}, R^2 = {R2_full:.6f}")
print("\nResumen del modelo (todos):")
print(model_full.summary())

# %%
print("\n(c) Ajuste y significancia excluyendo y=0 ...")
mask_nz = df[y_name] != 0
x_nz = df.loc[mask_nz, x_name].astype(float)
y_nz = df.loc[mask_nz, y_name].astype(float)
model_nz = fit_ols_xy(x_nz, y_nz)

print("Parámetros (sin y=0):")
print(model_nz.params)

MSE_nz = np.sum(model_nz.resid**2) / model_nz.df_resid
sigma_nz = np.sqrt(MSE_nz)
print(f"Sigma (EE de la regresión, sin y=0) = {sigma_nz:.6f}")

plt.figure(figsize=(7.2, 5.0))
plt.scatter(x_nz, y_nz, label="Datos (y>0)")
xx = np.linspace(x_nz.min(), x_nz.max(), 200)
X_line = sm.add_constant(xx)
y_hat = model_nz.predict(X_line)
plt.plot(xx, y_hat, label="Recta ajustada (y>0)")
plt.xlabel("Valor catastral (miles de pesos)")
plt.ylabel("Renta mensual (múltiplos de $5)")
plt.title("RLS (sin y=0): Renta ~ Valor")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plot_path2 = os.path.join(PLOTS_DIR, "nz_scatter_line.png")
plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
print("Figura guardada en:", plot_path2)
plt.show()

plt.figure(figsize=(7.2, 5.0))
plt.scatter(x_nz, model_nz.resid)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Valor catastral (miles de pesos)")
plt.ylabel("Residuales")
plt.title("Residuales vs x (sin y=0)")
plt.grid(True, linestyle="--", linewidth=0.5)
resid_path2 = os.path.join(PLOTS_DIR, "nz_resid_vs_x.png")
plt.savefig(resid_path2, dpi=300, bbox_inches="tight")
print("Figura guardada en:", resid_path2)
plt.show()

X_nz = sm.add_constant(x_nz)
model_nz = sm.OLS(y_nz, X_nz).fit()

SS_total_nz = np.sum((y_nz - np.mean(y_nz)) ** 2)
SS_model_nz = model_nz.ess
SS_resid_nz = model_nz.ssr
df_model_nz = model_nz.df_model
df_resid_nz = model_nz.df_resid
df_total_nz = df_model_nz + df_resid_nz

MS_model_nz = SS_model_nz / df_model_nz
MS_resid_nz = SS_resid_nz / df_resid_nz
F_stat_nz = MS_model_nz / MS_resid_nz
p_value_nz = 1 - stats.f.cdf(F_stat_nz, df_model_nz, df_resid_nz)

anova_data_nz = {
    "df": [df_model_nz, df_resid_nz, df_total_nz],
    "sum_sq": [SS_model_nz, SS_resid_nz, SS_total_nz],
    "mean_sq": [MS_model_nz, MS_resid_nz, np.nan],
    "F": [F_stat_nz, np.nan, np.nan],
    "PR(>F)": [p_value_nz, np.nan, np.nan],
}
anova_nz = pd.DataFrame(anova_data_nz, index=["x_valor_miles", "Residual", "Total"])
print("\nANOVA (sin y=0):\n", anova_nz.round(6))

F_nz = F_stat_nz
p_nz = p_value_nz
R2_nz = model_nz.rsquared
print(f"F = {F_nz:.6f}, p-valor = {p_nz:.6f}, R^2 = {R2_nz:.6f}")
print("\nResumen del modelo (sin y=0):")
print(model_nz.summary())

# %%
print("\n(d) Comparación de R^2 y guía de interpretación...")
print(f"R^2 (todos)   = {R2_full:.6f}")
print(f"R^2 (sin y=0) = {R2_nz:.6f}")
if R2_nz > R2_full:
    print("El ajuste mejora al remover y=0 (mayor R^2).")
elif R2_nz < R2_full:
    print("El ajuste empeora al remover y=0 (menor R^2).")
else:
    print("R^2 es igual en ambos casos.")
