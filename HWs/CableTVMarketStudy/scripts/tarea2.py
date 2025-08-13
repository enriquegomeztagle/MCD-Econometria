# %%
# %pip install pandas matplotlib

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
try:
    df = pd.read_excel("../data/cableTV.xlsx")
    print("File read successfully")
    read_success = True
except Exception as e:
    print(f"Error while reading the file: {e}")

# %%
if read_success:
    print("--------------------------------")
    print("Dataframe head:")
    print(df.head(10))

# %%
if read_success:
    print("--------------------------------")
    print("Dataframe info:")
    print(df.info())

# %%
if read_success:
    print("--------------------------------")
    print("Dataframe describe:")
    print(df.describe())

# %%
if read_success:
    print("--------------------------------")
    print("Dataframe shape:")
    print(df.shape)

# %%
if read_success:
    print("--------------------------------")
    print("Dataframe missing values:")
    print(df.isnull().sum())
    print("--------------------------------")
    print("Dataframe unique values:")
    print(df.nunique())

# %%
if read_success:
    print("--------------------------------")
    print("Dataframe unique values:")
    print(df.nunique())

# %%
quant_vars = ["adultos", "ninos", "teles", "tvtot", "renta", "valor"]
for var in quant_vars:
    plt.figure(figsize=(8, 5))
    values = df[var]
    label = var
    if var == "valor":
        values = values / 1000
        label += " (thousands of pesos)"
    plt.hist(values, bins="auto", edgecolor="black")
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.title(f"Frequency of {label}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../plots/python/freq_{var}.png", dpi=300, bbox_inches="tight")
# %%
cat_vars = ["colonia", "tipo"]
for var in cat_vars:
    print(f"=== Frequency of '{var}' ===")
    print(df[var].value_counts(dropna=False))
    print()

# %%
