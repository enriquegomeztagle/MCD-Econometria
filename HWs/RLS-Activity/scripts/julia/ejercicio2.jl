# Install required packages if not already installed
using Pkg
Pkg.add(["CSV", "DataFrames", "Plots", "GLM", "Distributions", "StatsBase"])

using DataFrames, CSV, Plots, Statistics, GLM, Distributions, StatsBase

DATA_PATH = "../../data/shampoo.csv"
PLOTS_DIR = "../../plots/julia/ejercicio2"
mkpath(PLOTS_DIR)

println("Cargando: $DATA_PATH")
df = CSV.read(DATA_PATH, DataFrame)
println("\nPrimeras filas:")
println(first(df, 5))

ventas_col_candidates = ["Ventas_Millones_lts", "Ventas", "ventas"]
inversion_col_candidates = ["Inversion_Millones_pesos", "Inversion", "inversion"]

ventas_col = first([c for c in ventas_col_candidates if c in names(df)])
inversion_col = first([c for c in inversion_col_candidates if c in names(df)])

x = convert.(Float64, df[:, inversion_col])
y = convert.(Float64, df[:, ventas_col])

n = length(x)
println("\nColumnas detectadas -> y: $ventas_col, x: $inversion_col; n = $n")

println("\n(a) Generando diagrama de dispersión y estadísticas descriptivas...")
scatter(x, y,
    xlabel="Inversión en redes (millones de pesos)",
    ylabel="Ventas (millones de litros)",
    title="Dispersión: Ventas vs. Inversión",
    grid=true, gridstyle=:dash, gridalpha=0.5,
    legend=false)

scatter_path = joinpath(PLOTS_DIR, "scatter_plot.png")
savefig(scatter_path)
println("Figura guardada en: $scatter_path")
display(current())

desc = DataFrame(
    "media" => [mean(y), mean(x)],
    "mediana" => [median(y), median(x)],
    "desv_est" => [std(y), std(x)],
    "min" => [minimum(y), minimum(x)],
    "Q1" => [quantile(y, 0.25), quantile(x, 0.25)],
    "Q3" => [quantile(y, 0.75), quantile(x, 0.75)],
    "max" => [maximum(y), maximum(x)],
    "variable" => ["Ventas (y)", "Inversión (x)"]
)
println("\nEstadísticas descriptivas:")
println(round.(desc[:, 1:7], digits=3))

println("\n(b) Correlación de Pearson e intervalo de confianza al 95%...")
r = cor(x, y)
println("r = $(round(r, digits=4))")

# Calcular p-valor manualmente para la correlación
t_stat = r * sqrt((n-2) / (1 - r^2))
pval = 2 * (1 - cdf(TDist(n-2), abs(t_stat)))
println("p-valor (bilateral) = $(round(pval, digits=6))")

z = atanh(r)
se_z = 1 / sqrt(n - 3)
z_crit = quantile(Normal(), 0.975)
lo_z, hi_z = z - z_crit * se_z, z + z_crit * se_z
lo_r, hi_r = tanh(lo_z), tanh(hi_z)
println("IC 95% para r: [$(round(lo_r, digits=4)), $(round(hi_r, digits=4))]")

println("\n(c) Ajustando RLS (MCO): y = beta0 + beta1*x + e ...")
model = lm(@formula(y ~ x), DataFrame(x=x, y=y))
println("\nParámetros estimados:")
println(coef(model))
println("\nResumen del modelo:")
println(model)

println("\n(d) Prueba de hipótesis (one-sided, alpha=0.05)...")

beta1_hat = coef(model)[2]
se_beta1 = stderror(model)[2]
df_resid = dof_residual(model)

beta1_H0 = 0.1

t_stat = (beta1_hat - beta1_H0) / se_beta1
p_one_sided = 1 - cdf(TDist(df_resid), t_stat)

println("beta1_hat = $(round(beta1_hat, digits=6))")
println("SE(beta1) = $(round(se_beta1, digits=6))")
println("t = $(round(t_stat, digits=4)), df = $df_resid")
println("p-valor (H1: beta1 > 0.1) = $(round(p_one_sided, digits=6))")

alpha = 0.05
if p_one_sided < alpha
    println("Conclusión: Se RECHAZA H0. La evidencia sugiere un incremento > 50 mil litros por cada 500 mil pesos.")
else
    println("Conclusión: NO se rechaza H0 al 5%. No hay evidencia suficiente para afirmar un incremento > 50 mil litros por cada 500 mil pesos.")
end

scatter(x, y, label="Datos")
xx = range(minimum(x), maximum(x), length=100)
X_line = [ones(length(xx)) xx]
y_hat = predict(model, DataFrame(x=xx))
plot!(xx, y_hat, label="Recta ajustada",
    xlabel="Inversión en redes (millones de pesos)",
    ylabel="Ventas (millones de litros)",
    title="RLS: Ventas ~ Inversión",
    grid=true, gridstyle=:dash, gridalpha=0.5)

line_path = joinpath(PLOTS_DIR, "shampoo_rls_line.png")
savefig(line_path)
println("Figura guardada en: $line_path")
display(current())


