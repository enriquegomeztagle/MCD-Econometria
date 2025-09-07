using Pkg
Pkg.add(["CSV", "DataFrames", "Plots", "GLM", "Distributions", "StatsBase"])
using DataFrames, CSV, Plots, Statistics, GLM, Distributions, StatsBase

df = CSV.read("../../data/datos_fuerza_levantamiento.csv", DataFrame)
first(df, 5)

println("Generando gráfico de dispersión...")
scatter(df.Fuerza_del_brazo_x, df.Levantamiento_dinamico_y,
    xlabel="Fuerza del brazo, x",
    ylabel="Levantamiento dinámico, y",
    title="Dispersión: y vs. x",
    grid=true, gridstyle=:dash, gridalpha=0.5,
    legend=false)

output_dir = "../../plots/julia/ejercicio1"
mkpath(output_dir)
savefig(joinpath(output_dir, "scatter_plot.png"))
display(current())


println("Generando estadísticas descriptivas...")
desc = describe(df[:, [:Fuerza_del_brazo_x, :Levantamiento_dinamico_y]])
println(desc)


println("Calculando coeficiente de correlación de Pearson...")
x = df.Fuerza_del_brazo_x
y = df.Levantamiento_dinamico_y

r = cor(x, y)
println("r (Pearson) = $(round(r, digits=4))")

n = length(x)
t_stat = r * sqrt((n-2) / (1 - r^2))
pval = 2 * (1 - cdf(TDist(n-2), abs(t_stat)))
println("p-valor = $(round(pval, digits=6))")

println("Realizando prueba de hipótesis...")
alpha = 0.05
if pval < alpha
    println("Conclusión: se rechaza H0; evidencia de relación lineal (α=0.05).")
else
    println("Conclusión: no se rechaza H0; no hay evidencia suficiente de relación lineal (α=0.05).")
end


model = lm(@formula(Levantamiento_dinamico_y ~ Fuerza_del_brazo_x), df)
println("Parámetros estimados:")
println(coef(model))

println("\nResumen del modelo:")
println(model)


println("Estimando modelo de regresión lineal simple por MCO...")
x0 = 30.0

X_new = [1.0 x0]
pred_mean = predict(model, DataFrame(Fuerza_del_brazo_x=[x0]))

n = length(x)
x̄ = mean(x)
Sxx = sum((x .- x̄).^2)
MSE = sum(residuals(model).^2) / (n - 2)

se_mean = sqrt(MSE * (1/n + (x0 - x̄)^2 / Sxx))
t_crit = quantile(TDist(n-2), 0.975)

ci_lower = pred_mean[1] - t_crit * se_mean
ci_upper = pred_mean[1] + t_crit * se_mean

se_pred = sqrt(MSE * (1 + 1/n + (x0 - x̄)^2 / Sxx))
pi_lower = pred_mean[1] - t_crit * se_pred
pi_upper = pred_mean[1] + t_crit * se_pred

println("Estimación puntual E[y|x=$x0] = $(round(pred_mean[1], digits=4))")
println("Intervalo de confianza al 95% para la media condicional:")
println("[$ci_lower, $ci_upper]")

println("\nIntervalo de predicción al 95% (para una observación nueva):")
println("[$pi_lower, $pi_upper]")


resid = residuals(model)
println("Generando gráfico de residuales...")
scatter(x, resid,
    xlabel="Fuerza del brazo, x",
    ylabel="Residuales",
    title="Residuales vs. x",
    grid=true, gridstyle=:dash, gridalpha=0.5,
    legend=false)
hline!([0], linestyle=:dash)

savefig("../../plots/julia/ejercicio1/residuals_plot.png")
display(current())

println("Media de residuales (debe ser ~0): $(round(mean(resid), digits=6))")
println("Desviación estándar de residuales: $(round(std(resid), digits=6))")
corr_rx = cor(x, resid)
println("Correlación(x, residuales) = $(round(corr_rx, digits=6)) (debe ser cercana a 0 en MCO con intercepto)")


