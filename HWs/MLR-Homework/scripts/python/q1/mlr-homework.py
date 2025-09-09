# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
data = pd.DataFrame({"Y": [1, 3, 8], "X2": [1, 2, 3], "X3": [2, 1, -3]})
print(data)

# %%
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
