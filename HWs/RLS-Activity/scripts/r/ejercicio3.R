# %%

library(readr)
library(ggplot2)
library(dplyr)
library(broom)
library(readxl)


DATA_CANDIDATES <- c(
  "../../data/flowers.xlsx",
  "../../data/flowers.csv"
)
PLOTS_DIR <- "../../plots/r/ejercicio3"
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
  stop("No se encontró ningún archivo de datos. Esperaba ../data/flowers.xlsx o ../data/flowers.csv")
}

cat("Cargando:", DATA_PATH, "\n")
if (grepl("\\.xlsx$", DATA_PATH)) {
  df <- read_excel(DATA_PATH)
} else {
  df <- read_csv(DATA_PATH)
}

cat("\nPrimeras filas:\n")
print(head(df))


x_candidates <- c("flores", "Flores", "X", "x")
y_candidates <- c("producción", "produccion", "Producción", "Y", "y")

x_col <- NULL
y_col <- NULL

for (col in x_candidates) {
  if (col %in% names(df)) {
    x_col <- col
    break
  }
}

for (col in y_candidates) {
  if (col %in% names(df)) {
    y_col <- col
    break
  }
}

x <- as.numeric(df[[x_col]])
y <- as.numeric(df[[y_col]])

n <- length(x)
cat("\nColumnas detectadas -> x:", x_col, ", y:", y_col, "; n =", n, "\n")

# %%
cat("\n(a) Gráfico de dispersión y estadísticas descriptivas...\n")


p <- ggplot(data.frame(x = x, y = y), aes(x = x, y = y)) +
  geom_point() +
  labs(x = "Flores procesadas, x (miles)",
       y = "Producción de esencia, y (onzas)",
       title = "Dispersión: Producción vs. Flores") +
  theme_minimal() +
  theme(panel.grid.minor = element_line(linetype = "dashed"))

scatter_path <- file.path(PLOTS_DIR, "scatter_flowers.png")
ggsave(scatter_path, p, dpi = 300, width = 6.8, height = 4.8)
cat("Figura guardada en:", scatter_path, "\n")
print(p)


desc_data <- data.frame(
  media = c(mean(x), mean(y)),
  mediana = c(median(x), median(y)),
  desv_est = c(sd(x), sd(y)),
  min = c(min(x), min(y)),
  Q1 = c(quantile(x, 0.25), quantile(y, 0.25)),
  Q3 = c(quantile(x, 0.75), quantile(y, 0.75)),
  max = c(max(x), max(y)),
  row.names = c("x (flores)", "y (producción)")
)
cat("\nEstadísticas descriptivas (redondeadas):\n")
print(round(desc_data, 3))

# %%
cat("\n(b) Relación lineal: signo y evidencia estadística...\n")
cor_result <- cor.test(x, y, method = "pearson")
r <- cor_result$estimate
pval <- cor_result$p.value

cat("Coef. de correlación de Pearson r =", round(r, 4), "\n")
cat("p-valor (bilateral) =", round(pval, 6), "\n")
if (r > 0) {
  cat("Dirección: positiva\n")
} else if (r < 0) {
  cat("Dirección: negativa\n")
} else {
  cat("Dirección: nula\n")
}

ci_cor <- cor_result$conf.int
cat("IC 95% de r: [", round(ci_cor[1], 4), ",", round(ci_cor[2], 4), "]\n")

# %%
cat("\n(c) Ajuste RLS (MCO) y verificación de b0, b1, S^2...\n")

model <- lm(y ~ x)
params <- coef(model)
b0_hat <- params[1]
b1_hat <- params[2]

resid <- residuals(model)
SSE <- sum(resid^2)
S2 <- SSE/(n-2)

cat("b0_hat =", round(b0_hat, 4), "\n")
cat("b1_hat =", round(b1_hat, 4), "\n")
cat("S^2 (SSE/(n-2)) =", round(S2, 4), "\n")
cat("\nResumen del modelo:\n")
print(summary(model))


p_line <- ggplot(data.frame(x = x, y = y), aes(x = x, y = y)) +
  geom_point(aes(color = "Datos")) +
  geom_smooth(method = "lm", se = FALSE, aes(color = "Recta ajustada")) +
  scale_color_manual(values = c("Datos" = "black", "Recta ajustada" = "red")) +
  labs(x = "Flores procesadas, x (miles)",
       y = "Producción de esencia, y (onzas)",
       title = "RLS: y ~ x",
       color = "Elementos") +
  theme_minimal() +
  theme(panel.grid.minor = element_line(linetype = "dashed"),
        legend.position = "bottom")

line_path <- file.path(PLOTS_DIR, "rls_line.png")
ggsave(line_path, p_line, dpi = 300, width = 6.8, height = 4.8)
cat("Figura guardada en:", line_path, "\n")
print(p_line)

# %%
cat("\n(c.1) Cálculos manuales para trazabilidad y verificación...\n")
Sxx <- sum((x - mean(x))^2)
Syy <- sum((y - mean(y))^2)
Sxy <- sum((x - mean(x))*(y - mean(y)))

b1_manual <- Sxy / Sxx
b0_manual <- mean(y) - b1_manual * mean(x)
SSE_manual <- sum((y - (b0_manual + b1_manual*x))^2)
S2_manual <- SSE_manual / (n - 2)

cat("Sxx =", round(Sxx, 6), ", Syy =", round(Syy, 6), ", Sxy =", round(Sxy, 6), "\n")
cat("b0_manual =", round(b0_manual, 4), ", b1_manual =", round(b1_manual, 4), "\n")
cat("S^2_manual =", round(S2_manual, 4), "\n")

b0_ref <- 1.38; b1_ref <- 0.52; S2_ref <- 0.206
cat("\nComparación con referencia (tolerancia +/- 0.03 para b0/b1 y +/- 0.02 para S^2):\n")
cat("|b0_manual -", b0_ref, "| =", round(abs(b0_manual - b0_ref), 4), "\n")
cat("|b1_manual -", b1_ref, "| =", round(abs(b1_manual - b1_ref), 4), "\n")
cat("|S^2_manual -", S2_ref, "| =", round(abs(S2_manual - S2_ref), 4), "\n")
cat("Coincide b0?", abs(b0_manual - b0_ref) <= 0.03, "\n")
cat("Coincide b1?", abs(b1_manual - b1_ref) <= 0.03, "\n")
cat("Coincide S^2?", abs(S2_manual - S2_ref) <= 0.02, "\n")

# %%
cat("\n(d) ANOVA de la regresión y prueba F...\n")
y_bar <- mean(y)
fitted_vals <- fitted(model)
SSR <- sum((fitted_vals - y_bar)^2)
SSE <- sum((y - fitted_vals)^2)
SST <- SSR + SSE

DF_model <- 1
DF_resid <- n - 2
DF_total <- n - 1

MSR <- SSR/DF_model
MSE <- SSE/DF_resid
F_stat <- MSR/MSE
p_F <- 1 - pf(F_stat, DF_model, DF_resid)

anova_table <- data.frame(
  SC = c(SSR, SSE, SST),
  gl = c(DF_model, DF_resid, DF_total),
  CM = c(MSR, MSE, NA),
  row.names = c("Regresión", "Error", "Total")
)
cat("Tabla ANOVA (redondeada):\n")
print(round(anova_table, 4))
cat("F =", round(F_stat, 4), ", df1 =", DF_model, ", df2 =", DF_resid, ", p-valor =", round(p_F, 6), "\n")

# %%
cat("\n(d.1) Diagnósticos del modelo...\n")
resid <- residuals(model)
fitted <- fitted_vals


p_resid_fit <- ggplot(data.frame(fitted = fitted, resid = resid), aes(x = fitted, y = resid)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed", linewidth = 1) +
  labs(x = "Valores ajustados",
       y = "Residuales",
       title = "Residuales vs. Ajustados") +
  theme_minimal() +
  theme(panel.grid.minor = element_line(linetype = "dashed", ))

resid_fit_path <- file.path(PLOTS_DIR, "residuals_vs_fitted.png")
ggsave(resid_fit_path, p_resid_fit, dpi = 300, width = 6.8, height = 4.6)
cat("Figura guardada en:", resid_fit_path, "\n")
print(p_resid_fit)

p_qq <- ggplot(data.frame(resid = resid), aes(sample = resid)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "QQ-plot de residuales") +
  theme_minimal()

qq_path <- file.path(PLOTS_DIR, "qqplot_residuals.png")
ggsave(qq_path, p_qq, dpi = 300, width = 6.8, height = 4.6)
cat("Figura guardada en:", qq_path, "\n")
print(p_qq)


p_hist <- ggplot(data.frame(resid = resid), aes(x = resid)) +
  geom_histogram(bins = 8, color = "black", fill = "white") +
  labs(title = "Histograma de residuales",
       x = "Residual",
       y = "Frecuencia") +
  theme_minimal()

hist_path <- file.path(PLOTS_DIR, "hist_residuals.png")
ggsave(hist_path, p_hist, dpi = 300, width = 6.8, height = 4.6)
cat("Figura guardada en:", hist_path, "\n")
print(p_hist)

skew <- moments::skewness(resid)
kurt <- moments::kurtosis(resid)
jb_stat <- n * (skew^2/6 + (kurt-3)^2/24)
jb_p <- 1 - pchisq(jb_stat, 2)
cat("Jarque-Bera: JB =", round(jb_stat, 4), ", p =", round(jb_p, 6), 
    ", skew =", round(skew, 4), ", kurt =", round(kurt, 4), "\n")

# %%
cat("\n(e) Error estándar de la pendiente e IC al 95%...\n")

model_summary <- summary(model)
se_b1 <- model_summary$coefficients[2, 2]

df_resid_model <- df.residual(model)
t_crit <- qt(0.975, df_resid_model)
ci_b1 <- c(b1_hat - t_crit*se_b1, b1_hat + t_crit*se_b1)
cat("SE(b1) =", round(se_b1, 6), "\n")
cat("IC 95% para b1: [", round(ci_b1[1], 4), ",", round(ci_b1[2], 4), "]\n")

# %%
cat("\n(f) Porcentaje de variabilidad explicada...\n")
R2 <- SSR / SST
cat("R^2 =", round(R2, 4), "->", round(100*R2, 2), "% de la variabilidad de y explicada por el modelo\n")

# %%
cat("\n(g) IC 95% para la media condicional en x0 = 1.25...\n")
x0 <- 1.25
new_data <- data.frame(x = x0)
pred_mean <- predict(model, new_data, interval = "confidence", level = 0.95)
cat("E[y|x0] puntual =", round(pred_mean[1], 4), "\n")
cat("IC 95% para E[y|x0]: [", round(pred_mean[2], 4), ",", round(pred_mean[3], 4), "]\n")

# %%
cat("\n(h) Intervalo de predicción al 95% en x0 = 1.95...\n")
x0 <- 1.95
new_data <- data.frame(x = x0)
pred_obs <- predict(model, new_data, interval = "prediction", level = 0.95)
cat("y_hat puntual =", round(pred_obs[1], 4), "\n")
cat("PI 95% para y nueva: [", round(pred_obs[2], 4), ",", round(pred_obs[3], 4), "]\n")

# %%
