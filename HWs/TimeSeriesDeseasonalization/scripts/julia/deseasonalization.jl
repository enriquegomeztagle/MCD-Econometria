using DataFrames, CSV, Statistics, GLM, Plots, StatsBase, StatsPlots, Distributions

df = CSV.read("../../data/avocado_exports.csv", DataFrame)
df.Año = Int.(df.Año)
df.Bimestre = Int.(df.Bimestre)
df.t = Int.(df.t)
df.Toneladas = Float64.(df.Toneladas)

media = mean(df.Toneladas)
desv = std(df.Toneladas)
cv = (desv / media) * 100
println("[1] Coeficiente de variación: $(round(cv, digits=2))%")

println("\n[2] Estadísticas descriptivas (Toneladas):")
println(describe(df.Toneladas))

p1 = plot(df.t, df.Toneladas, marker=:circle, 
          title="Serie bimestral de exportaciones de aguacate",
          xlabel="t (bimestres desde 2019-1)", ylabel="Toneladas",
          grid=true, legend=false)
savefig(p1, "../../plots/julia/avocado_exports_bimestral.png")
display(p1)

p2 = histogram(df.Toneladas, bins=10,
               title="Distribución de exportaciones (Toneladas)",
               xlabel="Toneladas", ylabel="Frecuencia",
               grid=true, legend=false)
savefig(p2, "../../plots/julia/avocado_exports_histogram.png")
display(p2)

p3 = boxplot(df.Toneladas, title="Boxplot de exportaciones bimestrales (Toneladas)",
              ylabel="Toneladas", grid=true, legend=false)
savefig(p3, "../../plots/julia/avocado_exports_boxplot.png")
display(p3)

X = [ones(size(df.t, 1)) df.t]
y = df.Toneladas
model_orig = lm(X, y)

pendiente = coef(model_orig)[2] / 1000.0
println("\n[3] Coeficiente de pendiente muestral (miles de toneladas por bimestre): $(round(pendiente, digits=4))")

println("\n[4] Prueba de hipótesis para pendiente > 0 (serie original)")
println(model_orig)

t_stat = coef(model_orig)[2] / stderror(model_orig)[2]
p_two_sided = 2 * (1 - cdf(TDist(dof_residual(model_orig)), abs(t_stat)))
p_one_sided = p_two_sided / 2
if t_stat < 0
    p_one_sided = 1 - p_one_sided
end

println("t = $(round(t_stat, digits=3)), p una-cola = $(round(p_one_sided, digits=4)) -> $(p_one_sided < 0.05 ? "Rechazamos H0: tendencia positiva" : "No se rechaza H0")")

k = 6
promedio_global = mean(df.Toneladas)
indices_estacionales = Dict()
for bim in 1:k
    indices_estacionales[bim] = mean(df.Toneladas[df.Bimestre .== bim]) / promedio_global
end

println("\n[5] Índices estacionales (multiplicativos):")
for bim in 1:k
    println("Bimestre $bim: $(round(indices_estacionales[bim], digits=4))")
end
println("Interpretación: valores >1 indican bimestres por encima del promedio; <1 por debajo.")

df.IndiceEstacional = [indices_estacionales[bim] for bim in df.Bimestre]
df.Deseasonalizada = df.Toneladas ./ df.IndiceEstacional

p4 = plot(df.t, df.Deseasonalizada, marker=:circle,
          title="Serie desestacionalizada (multiplicativa)",
          xlabel="t", ylabel="Toneladas desestacionalizadas",
          grid=true, legend=false)
savefig(p4, "../../plots/julia/avocado_exports_deseasonalized.png")
display(p4)

X_d = [ones(size(df.t, 1)) df.t]
y_d = df.Deseasonalizada
model_des = lm(X_d, y_d)

println("\n[6] Regresión con datos desestacionalizados:")
println(model_des)

t_stat_d = coef(model_des)[2] / stderror(model_des)[2]
p_two_sided_d = 2 * (1 - cdf(TDist(dof_residual(model_des)), abs(t_stat_d)))
p_one_sided_d = p_two_sided_d / 2
if t_stat_d < 0
    p_one_sided_d = 1 - p_one_sided_d
end

println("t = $(round(t_stat_d, digits=3)), p una-cola = $(round(p_one_sided_d, digits=4)) -> $(p_one_sided_d < 0.05 ? "Rechazamos H0: tendencia positiva" : "No se rechaza H0")")

t_future = 32
X_future = [1.0 t_future]
pred_mean = predict(model_orig, X_future)[1]
V = vcov(model_orig)
x0 = [1.0, t_future]
se_pred = sqrt(x0' * V * x0)
t_critical = quantile(TDist(dof_residual(model_orig)), 0.975)
ci_low = pred_mean - t_critical * se_pred
ci_high = pred_mean + t_critical * se_pred

println("\n[7] IC 95% para la tendencia (media esperada) en 2024-bim2 (t=32): [$(round(ci_low, digits=2)), $(round(ci_high, digits=2))] toneladas. Estimado: $(round(pred_mean, digits=2))")

r2_orig = r2(model_orig)
r2_des = r2(model_des)
println("\n[8] R^2 serie original: $(round(r2_orig, digits=4))")
println("[8] R^2 serie desestacionalizada: $(round(r2_des, digits=4))")
println("Explicación: al quitar la variación estacional, el componente sistemático por tiempo puede capturar mejor la tendencia subyacente (o a veces menos, si la estacionalidad ya explicaba variación alineada con t). Un cambio en R^2 refleja cuánto peso de la variabilidad se atribuye a la estacionalidad vs. la tendencia.")

fut = DataFrame(
    t = 31:36
)

function t_to_year_bim(t)
    base_year = 2019
    idx = t - 1
    year = base_year + div(idx, 6)
    bim = mod(idx, 6) + 1
    return year, bim
end

yb = [t_to_year_bim(t) for t in fut.t]
fut.Año = [y[1] for y in yb]
fut.Bimestre = [y[2] for y in yb]

X_fut = [ones(size(fut.t, 1)) fut.t]
pred_orig = predict(model_orig, X_fut)
fut.Pronostico_Original = pred_orig

pred_des = predict(model_des, X_fut)
fut.IndiceEstacional = [indices_estacionales[bim] for bim in fut.Bimestre]
fut.Pronostico_Deseason = pred_des .* fut.IndiceEstacional

println("\n[9] Pronósticos 2024 (toneladas):")
println(round.(fut[:, [:Año, :Bimestre, :t, :Pronostico_Original, :Pronostico_Deseason]], digits=2))

p5 = plot(df.t, df.Toneladas, marker=:circle, label="Observado")
plot!(fut.t, fut.Pronostico_Original, marker=:circle, linestyle=:dash, label="Pronóstico (original)")
title!("Pronóstico con modelo original")
xlabel!("t")
ylabel!("Toneladas")
plot!(grid=true)
savefig(p5, "../../plots/julia/avocado_exports_original_forecast.png")
display(p5)

p6 = plot(df.t, df.Toneladas, marker=:circle, label="Observado")
plot!(fut.t, fut.Pronostico_Deseason, marker=:circle, linestyle=:dash, label="Pronóstico (desestacionalizado)")
title!("Pronóstico con modelo desestacionalizado (reestacionalizado)")
xlabel!("t")
ylabel!("Toneladas")
plot!(grid=true)
savefig(p6, "../../plots/julia/avocado_exports_deseasonalized_forecast.png")
display(p6)
