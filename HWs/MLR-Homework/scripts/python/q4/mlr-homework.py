# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
raw = [
    (1947, 976.4, 1035.2, 5166.8, -10.351),
    (1948, 998.1, 1090.0, 5280.8, -4.720),
    (1949, 1025.3, 1095.6, 5607.4, 1.044),
    (1950, 1090.9, 1192.7, 5759.5, 0.407),
    (1951, 1107.7, 1227.0, 6081.6, -5.283),
    (1952, 1142.4, 1266.8, 6243.9, -0.277),
    (1953, 1221.4, 1327.5, 6355.6, 0.561),
    (1954, 1277.2, 1344.0, 6797.4, -0.138),
    (1955, 1314.0, 1433.8, 7172.2, 0.262),
    (1956, 1348.8, 1502.3, 7375.2, -0.736),
    (1957, 1381.8, 1539.5, 7315.3, -0.261),
    (1958, 1393.0, 1553.7, 7870.0, -0.575),
    (1959, 1470.7, 1623.8, 8188.1, 2.296),
    (1960, 1516.0, 1664.8, 8351.8, 1.511),
    (1961, 1541.2, 1720.0, 8971.9, 1.296),
    (1962, 1617.3, 1803.5, 9091.5, 1.396),
    (1963, 1684.8, 1871.5, 9436.1, 2.085),
    (1964, 1784.8, 2006.9, 10004.4, 2.027),
    (1965, 1897.6, 2131.0, 10562.8, 2.112),
    (1966, 2066.2, 2244.6, 11502.0, 2.220),
    (1967, 2066.2, 2340.5, 12341.0, 2.120),
    (1968, 2264.8, 2448.2, 12145.4, 1.055),
    (1969, 2314.5, 2524.3, 11672.3, 1.732),
    (1970, 2405.2, 2630.0, 11650.8, 1.176),
    (1971, 2505.5, 2745.3, 12312.9, -0.712),
    (1972, 2650.5, 2874.3, 13499.9, -0.156),
    (1973, 2675.9, 3072.3, 13081.0, 1.414),
    (1974, 2653.7, 3051.9, 11868.8, -1.043),
    (1975, 2710.9, 3108.5, 12634.4, -3.534),
    (1976, 2868.9, 3243.5, 13456.8, -0.657),
    (1977, 2992.1, 3360.7, 13786.3, -1.190),
    (1978, 3124.7, 3527.5, 14450.5, 0.113),
    (1979, 3203.2, 3628.6, 15340.0, 1.704),
    (1980, 3193.0, 3658.0, 15965.0, 2.298),
    (1981, 3236.0, 3741.1, 15965.0, 4.704),
    (1982, 3275.5, 3791.7, 16312.5, 4.449),
    (1983, 3454.3, 3906.9, 16944.8, 5.691),
    (1984, 3640.6, 4207.6, 17526.7, 5.848),
    (1985, 3820.9, 4347.8, 19068.3, 4.331),
    (1986, 3981.2, 4486.6, 20530.0, 3.768),
    (1987, 4113.4, 4586.5, 21235.7, 2.819),
    (1988, 4279.5, 4784.1, 22332.0, 3.287),
    (1989, 4393.7, 4906.5, 23659.8, 4.318),
    (1990, 4474.5, 5014.2, 23105.1, 3.595),
    (1991, 4466.6, 5033.0, 24050.2, 1.803),
    (1992, 4594.5, 5189.3, 24418.2, 1.007),
    (1993, 4748.9, 5261.3, 25092.3, 0.625),
    (1994, 4928.1, 5397.2, 25218.6, 2.206),
    (1995, 5075.6, 5539.1, 27439.7, 3.333),
    (1996, 5237.5, 5677.7, 29448.2, 3.083),
    (1997, 5423.9, 5854.5, 32664.1, 3.120),
    (1998, 5683.7, 6168.6, 35887.0, 3.584),
    (1999, 5968.4, 6320.0, 39591.3, 3.245),
    (2000, 6257.8, 6539.2, 38167.7, 3.576),
]
df = pd.DataFrame(raw, columns=["Año", "C", "Yd", "Riqueza", "Tasa"])
df.head()

# %%
X = sm.add_constant(df[["Yd", "Riqueza", "Tasa"]])
ols = sm.OLS(df["C"], X).fit()

print("=== Resumen statsmodels ===")
print(ols.summary())

# %%
b0, b1, b2, b3 = (
    ols.params["const"],
    ols.params["Yd"],
    ols.params["Riqueza"],
    ols.params["Tasa"],
)
print("\nEcuación ajustada:")
print(f"Ĉ_t = {b0:.3f} + {b1:.4f}·Yd_t + {b2:.4f}·Riqueza_t + {b3:.4f}·Tasa_t")

print(
    f"R² = {ols.rsquared:.4f}, R² ajustado = {ols.rsquared_adj:.4f}, n = {int(ols.nobs)}"
)

# %%
esperados = {"Yd": "pos", "Riqueza": "pos", "Tasa": "neg"}

print("\nSignos esperados vs. estimados:")
for var, signo in esperados.items():
    est = ols.params[var]
    cumple = est > 0 if signo == "pos" else est < 0
    print(f"{var}: coef = {est:.4f}, esperado {signo}, cumple? {cumple}")

# %%
stdY = df["C"].std()
stds = df[["Yd", "Riqueza", "Tasa"]].std()
betas_std = {
    "Yd": b1 * stds["Yd"] / stdY,
    "Riqueza": b2 * stds["Riqueza"] / stdY,
    "Tasa": b3 * stds["Tasa"] / stdY,
}
print("\nBetas estandarizados (para comparar magnitudes):")
for k, v in betas_std.items():
    print(f"{k}: {v:.4f}")

# %%
X_no_const = df[["Yd", "Riqueza", "Tasa"]].astype(float)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_no_const.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_no_const.values, i) for i in range(X_no_const.shape[1])
]
print("\nVIF (multicolinealidad, >10 es preocupante):")
print(vif_data.round(2))

# %%



