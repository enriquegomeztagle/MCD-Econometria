# Install required packages if not already installed
using Pkg
Pkg.add(["CSV", "DataFrames", "Plots", "GLM", "Distributions", "StatsBase", "XLSX"])

using DataFrames, CSV, Plots, Statistics, GLM, Distributions, StatsBase, XLSX

DATA_CANDIDATES = [
    "../../data/cableTV.xlsx",
    "../../data/cableTV.csv",
]
PLOTS_DIR = "../../plots/julia/ejercicio4"
mkpath(PLOTS_DIR)

global DATA_PATH = nothing
for path in DATA_CANDIDATES
    if isfile(path)
        global DATA_PATH = path
        break
    end
end

if DATA_PATH === nothing
    error("No se encontró cableTV.{xlsx,csv} en ../../data/")
end

println("Cargando: $DATA_PATH")
if endswith(DATA_PATH, ".xlsx")
    df = DataFrame(XLSX.readtable(DATA_PATH, "cableTV"))
else
    df = CSV.read(DATA_PATH, DataFrame)
end

println("\nColumnas disponibles: ", names(df))

rename!(df, lowercase.(names(df)))

expected = Set(["obs", "colonia", "manzana", "adultos", "ninos", "teles", "renta", "tvtot", "tipo", "valor"])
missing_cols = setdiff(expected, Set(names(df)))
if !isempty(missing_cols)
    println("Advertencia: faltan columnas esperadas: $missing_cols")
end

df.x_valor_miles = df.valor ./ 1000.0
x_name = :x_valor_miles
y_name = :renta
println("\nPrimeras filas:")
println(first(df[:, [:obs, :colonia, :manzana, :adultos, :ninos, :teles, y_name, :tvtot, :tipo, :valor, x_name]], 5))

function fit_ols_xy(x, y)
    model = lm(@formula(y ~ x), DataFrame(x=x, y=y))
    return model
end

println("\n(a) Ajuste MCO con todos los datos y gráficas...")
mask_full = ones(Bool, nrow(df))
x = convert.(Float64, df[:, x_name])
y = convert.(Float64, df[:, y_name])
model_full = fit_ols_xy(x, y)

println("\nParámetros (todos los datos):")
println(coef(model_full))

MSE_full = sum(residuals(model_full).^2) / dof_residual(model_full)
sigma_full = sqrt(MSE_full)
println("Sigma (EE de la regresión) = $(round(sigma_full, digits=6))")

scatter(x, y, label="Datos")
xx = range(minimum(x), maximum(x), length=200)
y_hat = predict(model_full, DataFrame(x=xx))
plot!(xx, y_hat, label="Recta ajustada",
    xlabel="Valor catastral (miles de pesos)",
    ylabel="Renta mensual (múltiplos de \$5)",
    title="RLS (todos): Renta ~ Valor",
    grid=true, gridstyle=:dash, gridalpha=0.5)

plot_path = joinpath(PLOTS_DIR, "full_scatter_line.png")
savefig(plot_path)
println("Figura guardada en: $plot_path")
display(current())

scatter(x, residuals(model_full),
    xlabel="Valor catastral (miles de pesos)",
    ylabel="Residuales",
    title="Residuales vs x (todos)",
    grid=true, gridstyle=:dash, gridalpha=0.5,
    legend=false)
hline!([0], linestyle=:dash, linewidth=1)

resid_path = joinpath(PLOTS_DIR, "full_resid_vs_x.png")
savefig(resid_path)
println("Figura guardada en: $resid_path")
display(current())

println("\n(b) ANOVA y significancia — todos los datos")
X_full = [ones(length(x)) x]
model_full = lm(@formula(y ~ x), DataFrame(x=x, y=y))

SS_total = sum((y .- mean(y)).^2)
SS_model = sum((fitted(model_full) .- mean(y)).^2)
SS_resid = sum(residuals(model_full).^2)
df_model = 1
df_resid = dof_residual(model_full)
df_total = df_model + df_resid

MS_model = SS_model / df_model
MS_resid = SS_resid / df_resid
F_stat = MS_model / MS_resid
p_value = 1 - cdf(FDist(df_model, df_resid), F_stat)

anova_data = DataFrame(
    "df" => [df_model, df_resid, df_total],
    "sum_sq" => [SS_model, SS_resid, SS_total],
    "mean_sq" => [MS_model, MS_resid, missing],
    "F" => [F_stat, missing, missing],
    "PR(>F)" => [p_value, missing, missing],
    "row" => ["x_valor_miles", "Residual", "Total"]
)
println("\nANOVA (todos):")
println(round.(anova_data[:, 1:5], digits=6))

F_full = F_stat
p_full = p_value
R2_full = r2(model_full)
println("F = $(round(F_full, digits=6)), p-valor = $(round(p_full, digits=6)), R^2 = $(round(R2_full, digits=6))")
println("\nResumen del modelo (todos):")
println(model_full)

println("\n(c) Ajuste y significancia excluyendo y=0 ...")
mask_nz = df[!, y_name] .!= 0
x_nz = convert.(Float64, df[mask_nz, x_name])
y_nz = convert.(Float64, df[mask_nz, y_name])
model_nz = fit_ols_xy(x_nz, y_nz)

println("Parámetros (sin y=0):")
println(coef(model_nz))

MSE_nz = sum(residuals(model_nz).^2) / dof_residual(model_nz)
sigma_nz = sqrt(MSE_nz)
println("Sigma (EE de la regresión, sin y=0) = $(round(sigma_nz, digits=6))")

scatter(x_nz, y_nz, label="Datos (y>0)")
xx = range(minimum(x_nz), maximum(x_nz), length=200)
y_hat = predict(model_nz, DataFrame(x=xx))
plot!(xx, y_hat, label="Recta ajustada (y>0)",
    xlabel="Valor catastral (miles de pesos)",
    ylabel="Renta mensual (múltiplos de \$5)",
    title="RLS (sin y=0): Renta ~ Valor",
    grid=true, gridstyle=:dash, gridalpha=0.5)

plot_path2 = joinpath(PLOTS_DIR, "nz_scatter_line.png")
savefig(plot_path2)
println("Figura guardada en: $plot_path2")
display(current())

scatter(x_nz, residuals(model_nz),
    xlabel="Valor catastral (miles de pesos)",
    ylabel="Residuales",
    title="Residuales vs x (sin y=0)",
    grid=true, gridstyle=:dash, gridalpha=0.5,
    legend=false)
hline!([0], linestyle=:dash, linewidth=1)

resid_path2 = joinpath(PLOTS_DIR, "nz_resid_vs_x.png")
savefig(resid_path2)
println("Figura guardada en: $resid_path2")
display(current())

X_nz = [ones(length(x_nz)) x_nz]
model_nz = lm(@formula(y ~ x), DataFrame(x=x_nz, y=y_nz))

SS_total_nz = sum((y_nz .- mean(y_nz)).^2)
SS_model_nz = sum((fitted(model_nz) .- mean(y_nz)).^2)
SS_resid_nz = sum(residuals(model_nz).^2)
df_model_nz = 1
df_resid_nz = dof_residual(model_nz)
df_total_nz = df_model_nz + df_resid_nz

MS_model_nz = SS_model_nz / df_model_nz
MS_resid_nz = SS_resid_nz / df_resid_nz
F_stat_nz = MS_model_nz / MS_resid_nz
p_value_nz = 1 - cdf(FDist(df_model_nz, df_resid_nz), F_stat_nz)

anova_data_nz = DataFrame(
    "df" => [df_model_nz, df_resid_nz, df_total_nz],
    "sum_sq" => [SS_model_nz, SS_resid_nz, SS_total_nz],
    "mean_sq" => [MS_model_nz, MS_resid_nz, missing],
    "F" => [F_stat_nz, missing, missing],
    "PR(>F)" => [p_value_nz, missing, missing],
    "row" => ["x_valor_miles", "Residual", "Total"]
)
println("\nANOVA (sin y=0):")
println(round.(anova_data_nz[:, 1:5], digits=6))

F_nz = F_stat_nz
p_nz = p_value_nz
R2_nz = r2(model_nz)
println("F = $(round(F_nz, digits=6)), p-valor = $(round(p_nz, digits=6)), R^2 = $(round(R2_nz, digits=6))")
println("\nResumen del modelo (sin y=0):")
println(model_nz)

println("\n(d) Comparación de R^2 y guía de interpretación...")
println("R^2 (todos)   = $(round(R2_full, digits=6))")
println("R^2 (sin y=0) = $(round(R2_nz, digits=6))")
if R2_nz > R2_full
    println("El ajuste mejora al remover y=0 (mayor R^2).")
elseif R2_nz < R2_full
    println("El ajuste empeora al remover y=0 (menor R^2).")
else
    println("R^2 es igual en ambos casos.")
end
