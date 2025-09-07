# %%

library(readr)
library(ggplot2)
library(dplyr)
library(broom)
library(readxl)


DATA_CANDIDATES <- c(
  "../../data/cableTV.xlsx",
  "../../data/cableTV.csv"
)
PLOTS_DIR <- "../../plots/r/ejercicio4"
if (!dir.exists(PLOTS_DIR)) {
  dir.create(PLOTS_DIR, recursive = TRUE)
}


DATA_PATH <- NULL
for (path in DATA_CANDIDATES) {
  if (file.exists(path)) {
    DATA_PATH <- path
    break
  }
}

if (is.null(DATA_PATH)) {
  stop("No se encontró cableTV.{xlsx,csv} en ../../data/")
}

cat("Cargando:", DATA_PATH, "\n")
if (grepl("\\.xlsx$", DATA_PATH)) {
  df <- read_excel(DATA_PATH)
} else {
  df <- read_csv(DATA_PATH)
}

cat("\nColumnas disponibles: ")
print(names(df))


names(df) <- tolower(names(df))

expected <- c("obs", "colonia", "manzana", "adultos", "ninos", "teles", "renta", "tvtot", "tipo", "valor")
missing_cols <- setdiff(expected, names(df))
if (length(missing_cols) > 0) {
  cat("Advertencia: faltan columnas esperadas:", missing_cols, "\n")
}

df$x_valor_miles <- df$valor / 1000.0
x_name <- "x_valor_miles"
y_name <- "renta"
cat("\nPrimeras filas:\n")
print(head(df[, c("obs", "colonia", "manzana", "adultos", "ninos", "teles", y_name, "tvtot", "tipo", "valor", x_name)]))

fit_ols_xy <- function(x, y) {
  model <- lm(y ~ x)
  return(model)
}

# %%
cat("\n(a) Ajuste MCO con todos los datos y gráficas...\n")
mask_full <- rep(TRUE, nrow(df))
x <- as.numeric(df[[x_name]])
y <- as.numeric(df[[y_name]])
model_full <- fit_ols_xy(x, y)

cat("\nParámetros (todos los datos):\n")
print(coef(model_full))

MSE_full <- sum(residuals(model_full)^2) / df.residual(model_full)
sigma_full <- sqrt(MSE_full)
cat("Sigma (EE de la regresión) =", round(sigma_full, 6), "\n")


p_full <- ggplot(data.frame(x = x, y = y), aes(x = x, y = y)) +
  geom_point(aes(color = "Datos")) +
  geom_smooth(method = "lm", se = FALSE, aes(color = "Recta ajustada")) +
  scale_color_manual(values = c("Datos" = "black", "Recta ajustada" = "red")) +
  labs(x = "Valor catastral (miles de pesos)",
       y = "Renta mensual (múltiplos de $5)",
       title = "RLS (todos): Renta ~ Valor",
       color = "Elementos") +
  theme_minimal() +
  theme(panel.grid.minor = element_line(linetype = "dashed"),
        legend.position = "bottom")

plot_path <- file.path(PLOTS_DIR, "full_scatter_line.png")
ggsave(plot_path, p_full, dpi = 300, width = 7.2, height = 5.0)
cat("Figura guardada en:", plot_path, "\n")
print(p_full)


p_resid_full <- ggplot(data.frame(x = x, resid = residuals(model_full)), aes(x = x, y = resid)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1) +
  labs(x = "Valor catastral (miles de pesos)",
       y = "Residuales",
       title = "Residuales vs x (todos)") +
  theme_minimal() +
  theme(panel.grid.minor = element_line(linetype = "dashed"))

resid_path <- file.path(PLOTS_DIR, "full_resid_vs_x.png")
ggsave(resid_path, p_resid_full, dpi = 300, width = 7.2, height = 5.0)
cat("Figura guardada en:", resid_path, "\n")
print(p_resid_full)

# %%
cat("\n(b) ANOVA y significancia — todos los datos\n")
X_full <- cbind(1, x)
model_full <- fit_ols_xy(x, y)


SS_total <- sum((y - mean(y))^2)
SS_model <- sum((fitted(model_full) - mean(y))^2)  
SS_resid <- sum(residuals(model_full)^2)  
df_model <- 1
df_resid <- df.residual(model_full)
df_total <- df_model + df_resid

MS_model <- SS_model / df_model
MS_resid <- SS_resid / df_resid
F_stat <- MS_model / MS_resid
p_value <- 1 - pf(F_stat, df_model, df_resid)


anova_data <- data.frame(
  df = c(df_model, df_resid, df_total),
  sum_sq = c(SS_model, SS_resid, SS_total),
  mean_sq = c(MS_model, MS_resid, NA),
  F = c(F_stat, NA, NA),
  PR_F = c(p_value, NA, NA),
  row.names = c("x_valor_miles", "Residual", "Total")
)
cat("\nANOVA (todos):\n")
print(round(anova_data, 6))

F_full <- F_stat
p_full <- p_value
R2_full <- summary(model_full)$r.squared
cat("F =", round(F_full, 6), ", p-valor =", round(p_full, 6), ", R^2 =", round(R2_full, 6), "\n")
cat("\nResumen del modelo (todos):\n")
print(summary(model_full))

# %%
cat("\n(c) Ajuste y significancia excluyendo y=0 ...\n")
mask_nz <- df[[y_name]] != 0
x_nz <- as.numeric(df[[x_name]][mask_nz])
y_nz <- as.numeric(df[[y_name]][mask_nz])
model_nz <- fit_ols_xy(x_nz, y_nz)

cat("Parámetros (sin y=0):\n")
print(coef(model_nz))

MSE_nz <- sum(residuals(model_nz)^2) / df.residual(model_nz)
sigma_nz <- sqrt(MSE_nz)
cat("Sigma (EE de la regresión, sin y=0) =", round(sigma_nz, 6), "\n")

p_nz <- ggplot(data.frame(x = x_nz, y = y_nz), aes(x = x, y = y)) +
  geom_point(aes(color = "Datos (y>0)")) +
  geom_smooth(method = "lm", se = FALSE, aes(color = "Recta ajustada (y>0)")) +
  scale_color_manual(values = c("Datos (y>0)" = "black", "Recta ajustada (y>0)" = "red")) +
  labs(x = "Valor catastral (miles de pesos)",
       y = "Renta mensual (múltiplos de $5)",
       title = "RLS (sin y=0): Renta ~ Valor",
       color = "Elementos") +
  theme_minimal() +
  theme(panel.grid.minor = element_line(linetype = "dashed"),
        legend.position = "bottom")

plot_path2 <- file.path(PLOTS_DIR, "nz_scatter_line.png")
ggsave(plot_path2, p_nz, dpi = 300, width = 7.2, height = 5.0)
cat("Figura guardada en:", plot_path2, "\n")
print(p_nz)

p_resid_nz <- ggplot(data.frame(x = x_nz, resid = residuals(model_nz)), aes(x = x, y = resid)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1) +
  labs(x = "Valor catastral (miles de pesos)",
       y = "Residuales",
       title = "Residuales vs x (sin y=0)") +
  theme_minimal() +
  theme(panel.grid.minor = element_line(linetype = "dashed"))

resid_path2 <- file.path(PLOTS_DIR, "nz_resid_vs_x.png")
ggsave(resid_path2, p_resid_nz, dpi = 300, width = 7.2, height = 5.0)
cat("Figura guardada en:", resid_path2, "\n")
print(p_resid_nz)

X_nz <- cbind(1, x_nz)
model_nz <- fit_ols_xy(x_nz, y_nz)


SS_total_nz <- sum((y_nz - mean(y_nz))^2)
SS_model_nz <- sum((fitted(model_nz) - mean(y_nz))^2)  
SS_resid_nz <- sum(residuals(model_nz)^2)  
df_model_nz <- 1
df_resid_nz <- df.residual(model_nz)
df_total_nz <- df_model_nz + df_resid_nz

MS_model_nz <- SS_model_nz / df_model_nz
MS_resid_nz <- SS_resid_nz / df_resid_nz
F_stat_nz <- MS_model_nz / MS_resid_nz
p_value_nz <- 1 - pf(F_stat_nz, df_model_nz, df_resid_nz)


anova_data_nz <- data.frame(
  df = c(df_model_nz, df_resid_nz, df_total_nz),
  sum_sq = c(SS_model_nz, SS_resid_nz, SS_total_nz),
  mean_sq = c(MS_model_nz, MS_resid_nz, NA),
  F = c(F_stat_nz, NA, NA),
  PR_F = c(p_value_nz, NA, NA),
  row.names = c("x_valor_miles", "Residual", "Total")
)
cat("\nANOVA (sin y=0):\n")
print(round(anova_data_nz, 6))

F_nz <- F_stat_nz
p_nz <- p_value_nz
R2_nz <- summary(model_nz)$r.squared
cat("F =", round(F_nz, 6), ", p-valor =", round(p_nz, 6), ", R^2 =", round(R2_nz, 6), "\n")
cat("\nResumen del modelo (sin y=0):\n")
print(summary(model_nz))

# %%
cat("\n(d) Comparación de R^2 y guía de interpretación...\n")
cat("R^2 (todos)   =", round(R2_full, 6), "\n")
cat("R^2 (sin y=0) =", round(R2_nz, 6), "\n")
if (R2_nz > R2_full) {
  cat("El ajuste mejora al remover y=0 (mayor R^2).\n")
} else if (R2_nz < R2_full) {
  cat("El ajuste empeora al remover y=0 (menor R^2).\n")
} else {
  cat("R^2 es igual en ambos casos.\n")
}

# %%
