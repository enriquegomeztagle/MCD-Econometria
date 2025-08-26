import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv("../../data/avocado_exports.csv")
df["Año"] = df["Año"].astype(int)
df["Bimestre"] = df["Bimestre"].astype(int)
df["t"] = df["t"].astype(int)
df["Toneladas"] = df["Toneladas"].astype(float)

media = df["Toneladas"].mean()
desv = df["Toneladas"].std(ddof=1)
cv = (desv / media) * 100
print(f"[1] Coeficiente de variación: {cv:.2f}%")

print("\n[2] Estadísticas descriptivas (Toneladas):")
print(df["Toneladas"].describe())

plt.figure()
plt.plot(df["t"], df["Toneladas"], marker="o")
plt.title("Serie bimestral de exportaciones de aguacate")
plt.xlabel("t (bimestres desde 2019-1)")
plt.ylabel("Toneladas")
plt.grid(True)
plt.savefig("../../plots/python/avocado_exports_bimestral.png")
plt.show()

plt.figure()
plt.hist(df["Toneladas"], bins=10)
plt.title("Distribución de exportaciones (Toneladas)")
plt.xlabel("Toneladas")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.savefig("../../plots/python/avocado_exports_histogram.png")
plt.show()

plt.figure()
plt.boxplot(df["Toneladas"], vert=True, showmeans=True)
plt.title("Boxplot de exportaciones bimestrales (Toneladas)")
plt.ylabel("Toneladas")
plt.grid(True)
plt.savefig("../../plots/python/avocado_exports_boxplot.png")
plt.show()

X = sm.add_constant(df["t"])
y = df["Toneladas"]
model_orig = sm.OLS(y, X).fit()

pendiente = model_orig.params["t"] / 1000.0
print(
    f"\n[3] Coeficiente de pendiente muestral (miles de toneladas por bimestre): {pendiente:.4f}"
)

print("\n[4] Prueba de hipótesis para pendiente > 0 (serie original)")
print(model_orig.summary())

t_stat = model_orig.tvalues["t"]
p_two_sided = model_orig.pvalues["t"]
p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
print(
    f"t = {t_stat:.3f}, p una-cola = {p_one_sided:.4g}  -> "
    f"{'Rechazamos H0: tendencia positiva' if p_one_sided < 0.05 else 'No se rechaza H0'}"
)

k = 6
promedio_global = df["Toneladas"].mean()
indices_estacionales = df.groupby("Bimestre")["Toneladas"].mean() / promedio_global
indices_estacionales = indices_estacionales.reindex(range(1, k + 1))

print("\n[5] Índices estacionales (multiplicativos):")
for bim, ind in indices_estacionales.items():
    print(f"Bimestre {bim}: {ind:.4f}")
print(
    "Interpretación: valores >1 indican bimestres por encima del promedio; <1 por debajo."
)

df["IndiceEstacional"] = df["Bimestre"].map(indices_estacionales)
df["Deseasonalizada"] = df["Toneladas"] / df["IndiceEstacional"]

plt.figure()
plt.plot(df["t"], df["Deseasonalizada"], marker="o")
plt.title("Serie desestacionalizada (multiplicativa)")
plt.xlabel("t")
plt.ylabel("Toneladas desestacionalizadas")
plt.grid(True)
plt.savefig("../../plots/python/avocado_exports_deseasonalized.png")
plt.show()

X_d = sm.add_constant(df["t"])
y_d = df["Deseasonalizada"]
model_des = sm.OLS(y_d, X_d).fit()
print("\n[6] Regresión con datos desestacionalizados:")
print(model_des.summary())

t_stat_d = model_des.tvalues["t"]
p_two_sided_d = model_des.pvalues["t"]
p_one_sided_d = p_two_sided_d / 2 if t_stat_d > 0 else 1 - p_two_sided_d / 2
print(
    f"t = {t_stat_d:.3f}, p una-cola = {p_one_sided_d:.4g}  -> "
    f"{'Rechazamos H0: tendencia positiva' if p_one_sided_d < 0.05 else 'No se rechaza H0'}"
)

t_future = 32
X_future = pd.DataFrame({"const": 1.0, "t": [t_future]})
pred_mean = model_orig.get_prediction(X_future)
ci_low, ci_high = pred_mean.conf_int(alpha=0.05)[0]
mean_pred = pred_mean.predicted_mean[0]
print(
    f"\n[7] IC 95% para la tendencia (media esperada) en 2024-bim2 (t=32): "
    f"[{ci_low:.2f}, {ci_high:.2f}] toneladas. Estimado: {mean_pred:.2f}"
)
pi = pred_mean.summary_frame(alpha=0.05)[["obs_ci_lower", "obs_ci_upper"]].values[0]
print(
    f"[7-extra] IC 95% de predicción (valor observado esperado) en 2024-bim2: [{pi[0]:.2f}, {pi[1]:.2f}]"
)

r2_orig = model_orig.rsquared
r2_des = model_des.rsquared
print(f"\n[8] R^2 serie original: {r2_orig:.4f}")
print(f"[8] R^2 serie desestacionalizada: {r2_des:.4f}")
print(
    "Explicación: al quitar la variación estacional, el componente sistemático por tiempo "
    "puede capturar mejor la tendencia subyacente (o a veces menos, si la estacionalidad "
    "ya explicaba variación alineada con t). Un cambio en R^2 refleja cuánto peso de la "
    "variabilidad se atribuye a la estacionalidad vs. la tendencia."
)

fut = pd.DataFrame(
    {
        "t": np.arange(31, 37),
    }
)


def t_to_year_bim(t):
    base_year = 2019
    idx = t - 1
    year = base_year + idx // 6
    bim = (idx % 6) + 1
    return year, bim


yb = np.array([t_to_year_bim(t) for t in fut["t"]])
fut["Año"] = yb[:, 0]
fut["Bimestre"] = yb[:, 1]
fut["const"] = 1.0

pred_orig = model_orig.get_prediction(fut[["const", "t"]])
fut["Pronostico_Original"] = pred_orig.predicted_mean

pred_des = model_des.get_prediction(fut[["const", "t"]]).predicted_mean
fut["IndiceEstacional"] = fut["Bimestre"].map(indices_estacionales)
fut["Pronostico_Deseason"] = pred_des * fut["IndiceEstacional"]

print("\n[9] Pronósticos 2024 (toneladas):")
print(
    fut[["Año", "Bimestre", "t", "Pronostico_Original", "Pronostico_Deseason"]].round(2)
)

pred_orig_full = model_orig.get_prediction(fut[["const", "t"]])
pred_des_full = model_des.get_prediction(fut[["const", "t"]])

fut["IC_orig_low"], fut["IC_orig_high"] = pred_orig_full.conf_int(alpha=0.05).T
pi_des = pred_des_full.summary_frame(alpha=0.05)
fut["IC_des_low"], fut["IC_des_high"] = (
    pi_des["obs_ci_lower"] * fut["IndiceEstacional"],
    pi_des["obs_ci_upper"] * fut["IndiceEstacional"],
)

print("\n[9-extra] Pronósticos con intervalos de confianza:")
print(
    fut[
        [
            "Año",
            "Bimestre",
            "Pronostico_Original",
            "IC_orig_low",
            "IC_orig_high",
            "Pronostico_Deseason",
            "IC_des_low",
            "IC_des_high",
        ]
    ].round(2)
)

plt.figure()
plt.plot(df["t"], df["Toneladas"], marker="o", label="Observado")
plt.plot(
    fut["t"],
    fut["Pronostico_Original"],
    marker="o",
    linestyle="--",
    label="Pronóstico (original)",
)
plt.title("Pronóstico con modelo original")
plt.xlabel("t")
plt.ylabel("Toneladas")
plt.legend()
plt.grid(True)
plt.savefig("../../plots/python/avocado_exports_original_forecast.png")
plt.show()

plt.figure()
plt.plot(df["t"], df["Toneladas"], marker="o", label="Observado")
plt.plot(
    fut["t"],
    fut["Pronostico_Deseason"],
    marker="o",
    linestyle="--",
    label="Pronóstico (desestacionalizado)",
)
plt.title("Pronóstico con modelo desestacionalizado (reestacionalizado)")
plt.xlabel("t")
plt.ylabel("Toneladas")
plt.legend()
plt.grid(True)
plt.savefig("../../plots/python/avocado_exports_deseasonalized_forecast.png")
plt.show()
