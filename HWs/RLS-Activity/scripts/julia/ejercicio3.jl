
# Install required packages if not already installed
using Pkg
Pkg.add(["CSV", "DataFrames", "Plots", "GLM", "Distributions", "StatsBase", "XLSX"])

using DataFrames, CSV, Plots, Statistics, GLM, Distributions, StatsBase, XLSX

DATA_CANDIDATES = [
    "../../data/flowers.xlsx",
    "../../data/flowers.csv",
]
PLOTS_DIR = "../../plots/julia/ejercicio3"
mkpath(PLOTS_DIR)

global DATA_PATH = nothing
for path in DATA_CANDIDATES
    if isfile(path)
        global DATA_PATH = path
        break
    end
end

if DATA_PATH === nothing
    error("No se encontró ningún archivo de datos. Esperaba ../../data/flowers.xlsx o ../../data/flowers.csv")
end

println("Cargando: $DATA_PATH")
if endswith(DATA_PATH, ".xlsx")
    df = DataFrame(XLSX.readtable(DATA_PATH, "flores"))
else
    df = CSV.read(DATA_PATH, DataFrame)
end

println("\nPrimeras filas:")
println(first(df, 5))

x_candidates = ["flores", "Flores", "X", "x"]
y_candidates = ["producción", "produccion", "Producción", "Y", "y"]

x_col = first([c for c in x_candidates if c in names(df)])
y_col = first([c for c in y_candidates if c in names(df)])

x = convert.(Float64, df[:, x_col])
y = convert.(Float64, df[:, y_col])

n = length(x)
println("\nColumnas detectadas -> x: $x_col, y: $y_col; n = $n")


println("\n(a) Gráfico de dispersión y estadísticas descriptivas...")
scatter(x, y,
    xlabel="Flores procesadas, x (miles)",
    ylabel="Producción de esencia, y (onzas)",
    title="Dispersión: Producción vs. Flores",
    grid=true, gridstyle=:dash, gridalpha=0.5,
    legend=false)

scatter_path = joinpath(PLOTS_DIR, "scatter_flowers.png")
savefig(scatter_path)
println("Figura guardada en: $scatter_path")
display(current())

desc = DataFrame(
    "media" => [mean(x), mean(y)],
    "mediana" => [median(x), median(y)],
    "desv_est" => [std(x), std(y)],
    "min" => [minimum(x), minimum(y)],
    "Q1" => [quantile(x, 0.25), quantile(y, 0.25)],
    "Q3" => [quantile(x, 0.75), quantile(y, 0.75)],
    "max" => [maximum(x), maximum(y)],
    "variable" => ["x (flores)", "y (producción)"]
)
println("\nEstadísticas descriptivas (redondeadas):")
println(round.(desc[:, 1:7], digits=3))


println("\n(b) Relación lineal: signo y evidencia estadística...")
r = cor(x, y)
println("Coef. de correlación de Pearson r = $(round(r, digits=4))")

# Calcular p-valor manualmente para la correlación
t_stat = r * sqrt((n-2) / (1 - r^2))
pval = 2 * (1 - cdf(TDist(n-2), abs(t_stat)))
println("p-valor (bilateral) = $(round(pval, digits=6))")
if r > 0
    println("Dirección: positiva")
elseif r < 0
    println("Dirección: negativa")
else
    println("Dirección: nula")
end

z = atanh(r)
se_z = 1/sqrt(n-3)
z_crit = quantile(Normal(), 0.975)
lo_r, hi_r = tanh(z - z_crit*se_z), tanh(z + z_crit*se_z)
println("IC 95% de r: [$(round(lo_r, digits=4)), $(round(hi_r, digits=4))]")


println("\n(c) Ajuste RLS (MCO) y verificación de b0, b1, S^2...")
model = lm(@formula(y ~ x), DataFrame(x=x, y=y))
params = coef(model)
b0_hat = params[1]
b1_hat = params[2]

resid = residuals(model)
SSE = sum(resid.^2)
S2 = SSE/(n-2)

println("b0_hat = $(round(b0_hat, digits=4))")
println("b1_hat = $(round(b1_hat, digits=4))")
println("S^2 (SSE/(n-2)) = $(round(S2, digits=4))")
println("\nResumen del modelo:")
println(model)

scatter(x, y, label="Datos")
xx = range(minimum(x), maximum(x), length=100)
y_hat = predict(model, DataFrame(x=xx))
plot!(xx, y_hat, label="Recta ajustada",
    xlabel="Flores procesadas, x (miles)",
    ylabel="Producción de esencia, y (onzas)",
    title="RLS: y ~ x",
    grid=true, gridstyle=:dash, gridalpha=0.5)

line_path = joinpath(PLOTS_DIR, "rls_line.png")
savefig(line_path)
println("Figura guardada en: $line_path")
display(current())


println("\n(c.1) Cálculos manuales para trazabilidad y verificación...")
Sxx = sum((x .- mean(x)).^2)
Syy = sum((y .- mean(y)).^2)
Sxy = sum((x .- mean(x)).*(y .- mean(y)))

b1_manual = Sxy / Sxx
b0_manual = mean(y) - b1_manual * mean(x)
SSE_manual = sum((y .- (b0_manual .+ b1_manual.*x)).^2)
S2_manual = SSE_manual / (n - 2)

println("Sxx = $(round(Sxx, digits=6)), Syy = $(round(Syy, digits=6)), Sxy = $(round(Sxy, digits=6))")
println("b0_manual = $(round(b0_manual, digits=4)), b1_manual = $(round(b1_manual, digits=4))")
println("S^2_manual = $(round(S2_manual, digits=4))")

b0_ref, b1_ref, S2_ref = 1.38, 0.52, 0.206
println("\nComparación con referencia (tolerancia +/- 0.03 para b0/b1 y +/- 0.02 para S^2):")
println("|b0_manual - $b0_ref| = $(round(abs(b0_manual - b0_ref), digits=4))")
println("|b1_manual - $b1_ref| = $(round(abs(b1_manual - b1_ref), digits=4))")
println("|S^2_manual - $S2_ref| = $(round(abs(S2_manual - S2_ref), digits=4))")
println("Coincide b0? ", abs(b0_manual - b0_ref) <= 0.03)
println("Coincide b1? ", abs(b1_manual - b1_ref) <= 0.03)
println("Coincide S^2? ", abs(S2_manual - S2_ref) <= 0.02)


println("\n(d) ANOVA de la regresión y prueba F...")
y_bar = mean(y)
fitted_vals = predict(model)
SSR = sum((fitted_vals .- y_bar).^2)
SSE = sum((y .- fitted_vals).^2)
SST = SSR + SSE

DF_model = 1
DF_resid = n - 2
DF_total = n - 1

MSR = SSR/DF_model
MSE = SSE/DF_resid
F_stat = MSR/MSE
p_F = 1 - cdf(FDist(DF_model, DF_resid), F_stat)

anova = DataFrame(
    "SC" => [SSR, SSE, SST],
    "gl" => [DF_model, DF_resid, DF_total],
    "CM" => [MSR, MSE, missing],
    "row" => ["Regresión", "Error", "Total"]
)
println("Tabla ANOVA (redondeada):")
println(round.(anova[:, 1:3], digits=4))
println("F = $(round(F_stat, digits=4)), df1 = $DF_model, df2 = $DF_resid, p-valor = $(round(p_F, digits=6))")


println("\n(d.1) Diagnósticos del modelo...")
resid = residuals(model)
fitted = fitted_vals

scatter(fitted, resid,
    xlabel="Valores ajustados",
    ylabel="Residuales",
    title="Residuales vs. Ajustados",
    grid=true, gridstyle=:dash, gridalpha=0.5,
    legend=false)
hline!([0], color=:black, linestyle=:dash, linewidth=1)

resid_fit_path = joinpath(PLOTS_DIR, "residuals_vs_fitted.png")
savefig(resid_fit_path)
println("Figura guardada en: $resid_fit_path")
display(current())

# QQ-plot usando cuantiles teóricos vs empíricos
q_theoretical = quantile.(Normal(), range(0.01, 0.99, length=100))
q_empirical = quantile(resid, range(0.01, 0.99, length=100))

qq_plot = scatter(q_theoretical, q_empirical,
    xlabel="Cuantiles teóricos",
    ylabel="Cuantiles de residuales",
    title="QQ-plot de residuales",
    grid=true, gridstyle=:dash, gridalpha=0.5,
    legend=false)
plot!([minimum(q_theoretical), maximum(q_theoretical)], [minimum(q_theoretical), maximum(q_theoretical)], 
    color=:red, linestyle=:dash, label="Línea de referencia")

qq_path = joinpath(PLOTS_DIR, "qqplot_residuals.png")
savefig(qq_path)
println("Figura guardada en: $qq_path")
display(current())

histogram(resid, bins=8, edgecolor=:black,
    title="Histograma de residuales",
    xlabel="Residual",
    ylabel="Frecuencia",
    legend=false)
hist_path = joinpath(PLOTS_DIR, "hist_residuals.png")
savefig(hist_path)
println("Figura guardada en: $hist_path")
display(current())

skew = skewness(resid)
kurt = kurtosis(resid)
jb_stat = n * (skew^2/6 + (kurt-3)^2/24)
jb_p = 1 - cdf(Chisq(2), jb_stat)
println("Jarque-Bera: JB = $(round(jb_stat, digits=4)), p = $(round(jb_p, digits=6)), skew = $(round(skew, digits=4)), kurt = $(round(kurt, digits=4))")


println("\n(e) Error estándar de la pendiente e IC al 95%...")
Sxx = sum((x .- mean(x)).^2)
se_b1 = sqrt(MSE / Sxx)
t_crit = quantile(TDist(DF_resid), 0.975)
ci_b1 = (b1_hat - t_crit*se_b1, b1_hat + t_crit*se_b1)
println("SE(b1) = $(round(se_b1, digits=6))")
println("IC 95% para b1: [$(round(ci_b1[1], digits=4)), $(round(ci_b1[2], digits=4))]")


println("\n(f) Porcentaje de variabilidad explicada...")
R2 = SSR / SST
println("R^2 = $(round(R2, digits=4)) -> $(round(100*R2, digits=2))% de la variabilidad de y explicada por el modelo")


println("\n(g) IC 95% para la media condicional en x0 = 1.25...")
x0 = 1.25
pred_mean = predict(model, DataFrame(x=[x0]))
se_mean = sqrt(MSE * (1/n + (x0 - mean(x))^2 / Sxx))
mean_hat = pred_mean[1]
ci_mean = (mean_hat - t_crit*se_mean, mean_hat + t_crit*se_mean)
println("E[y|x0] puntual = $(round(mean_hat, digits=4))")
println("IC 95% para E[y|x0]: [$(round(ci_mean[1], digits=4)), $(round(ci_mean[2], digits=4))]")


println("\n(h) Intervalo de predicción al 95% en x0 = 1.95...")
x0 = 1.95
pred_obs = predict(model, DataFrame(x=[x0]))
se_pred = sqrt(MSE * (1 + 1/n + (x0 - mean(x))^2 / Sxx))
mean_hat = pred_obs[1]
pi = (mean_hat - t_crit*se_pred, mean_hat + t_crit*se_pred)
println("y_hat puntual = $(round(mean_hat, digits=4))")
println("PI 95% para y nueva: [$(round(pi[1], digits=4)), $(round(pi[2], digits=4))]")


