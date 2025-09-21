# %%
from scipy.stats import f

# %%
SCR1, df1 = 55.0, 25
SCR2, df2 = 140.0, 25
alpha = 0.05

# %%
F = (SCR2 / df2) / (SCR1 / df1)
F_crit_upper = f.ppf(1 - alpha, df2, df1)
p_one_sided = f.sf(F, df2, df1)
p_two_sided = 2 * min(f.sf(F, df2, df1), f.sf(1.0 / F, df1, df2))
p_two_sided = min(p_two_sided, 1.0)

# %%
print(f"F observado = {F:.6f}")
print(f"F crítico (una cola, α={alpha}, gl=({df2},{df1})) = {F_crit_upper:.6f}")
print(f"p-value (una cola) = {p_one_sided:.6f}")
print(f"p-value (dos colas) = {p_two_sided:.6f}")

# %%
if F > F_crit_upper:
    print(
        "Conclusión (una cola, 5%): Rechaza H0 de homoscedasticidad (evidencia de heteroscedasticidad)."
    )
else:
    print("Conclusión (una cola, 5%): No se rechaza H0 de homoscedasticidad.")

if p_two_sided < alpha:
    print("Conclusión (dos colas, 5%): Rechaza H0 de homoscedasticidad.")
else:
    print("Conclusión (dos colas, 5%): No se rechaza H0 de homoscedasticidad.")

# %%
