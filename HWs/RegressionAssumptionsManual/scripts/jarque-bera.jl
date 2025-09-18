# Install packages
# using Pkg
# Pkg.add(["Random", "DataFrames", "GLM", "HypothesisTests", "Statistics"])

using Random, DataFrames, GLM, HypothesisTests, Statistics
using HypothesisTests: pvalue

Random.seed!(42)
n = 300
variable_independiente = randn(n)
variable_dependiente   = 5 .+ 2 .* randn(n) .+ randn(n)

df = DataFrame(variable_dependiente = variable_dependiente,
               variable_independiente = variable_independiente)

# OLS
modelo = lm(@formula(variable_dependiente ~ variable_independiente), df)
residuos = residuals(modelo)

# Jarque–Bera
alpha = 0.05
t = JarqueBeraTest(residuos)
jb_stat = t.JB
p = pvalue(t)

println("JB: ", round(jb_stat, digits=4))
println("p-valor: ", round(p, digits=6))
if p < alpha
    println("Rechazamos H0: los residuos no son normales.")
else
    println("No rechazamos H0: no hay evidencia contra la normalidad.")
end
