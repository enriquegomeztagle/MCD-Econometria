# %%
library(readr)
library(ggplot2)
library(dplyr)
library(broom)

DATA_PATH <- "../../data/shampoo.csv"
PLOTS_DIR <- "../../plots/r/ejercicio2"
if (!dir.exists(PLOTS_DIR)) {
  dir.create(PLOTS_DIR, recursive = TRUE)
}

cat("Cargando:", DATA_PATH, "\n")
df <- read_csv(DATA_PATH)
cat("\nPrimeras filas:\n")
print(head(df))

ventas_col_candidates <- c("Ventas_Millones_lts", "Ventas", "ventas")
inversion_col_candidates <- c("Inversion_Millones_pesos", "Inversion", "inversion")

ventas_col <- NULL
inversion_col <- NULL

for (col in ventas_col_candidates) {
  if (col %in% names(df)) {
    ventas_col <- col
    break
  }
}

for (col in inversion_col_candidates) {
  if (col %in% names(df)) {
    inversion_col <- col
    break
  }
}

x <- as.numeric(df[[inversion_col]])
y <- as.numeric(df[[ventas_col]])

n <- length(x)
cat("\nColumnas detectadas -> y:", ventas_col, ", x:", inversion_col, "; n =", n, "\n")

# %%
cat("\n(a) Generando diagrama de dispersión y estadísticas descriptivas...\n")

p <- ggplot(data.frame(x = x, y = y), aes(x = x, y = y)) +
  geom_point() +
  labs(x = "Inversión en redes (millones de pesos)",
       y = "Ventas (millones de litros)",
       title = "Dispersión: Ventas vs. Inversión") +
  theme_minimal() +
  theme(panel.grid.minor = element_line(linetype = "dashed"))

scatter_path <- file.path(PLOTS_DIR, "scatter_plot.png")
ggsave(scatter_path, p, dpi = 300, width = 6.5, height = 4.5)
cat("Figura guardada en:", scatter_path, "\n")
print(p)

desc_data <- data.frame(
  media = c(mean(y), mean(x)),
  mediana = c(median(y), median(x)),
  desv_est = c(sd(y), sd(x)),
  min = c(min(y), min(x)),
  Q1 = c(quantile(y, 0.25), quantile(x, 0.25)),
  Q3 = c(quantile(y, 0.75), quantile(x, 0.75)),
  max = c(max(y), max(x)),
  row.names = c("Ventas (y)", "Inversión (x)")
)

cat("\nEstadísticas descriptivas:\n")
print(round(desc_data, 3))

# %%
cat("\n(b) Correlación de Pearson e intervalo de confianza al 95%...\n")
cor_result <- cor.test(x, y, method = "pearson")
r <- cor_result$estimate
pval <- cor_result$p.value

cat("r =", round(r, 4), "\n")
cat("p-valor (bilateral) =", round(pval, 6), "\n")

ci_cor <- cor_result$conf.int
cat("IC 95% para r (cor.test): [", round(ci_cor[1], 4), ",", round(ci_cor[2], 4), "]\n")

# %%
cat("\n(c) Ajustando RLS (MCO): y = beta0 + beta1*x + e ...\n")
model <- lm(y ~ x)
cat("\nParámetros estimados:\n")
print(coef(model))
cat("\nResumen del modelo:\n")
print(summary(model))

# %%
cat("\n(d) Prueba de hipótesis (one-sided, alpha=0.05)...\n")

beta1_hat <- coef(model)[2]
se_beta1 <- summary(model)$coefficients[2, 2]
df_resid <- df.residual(model)

beta1_H0 <- 0.1

t_stat <- (beta1_hat - beta1_H0) / se_beta1
p_one_sided <- 1 - pt(t_stat, df_resid)

cat("beta1_hat =", round(beta1_hat, 6), "\n")
cat("SE(beta1) =", round(se_beta1, 6), "\n")
cat("t =", round(t_stat, 4), ", df =", df_resid, "\n")
cat("p-valor (H1: beta1 > 0.1) =", round(p_one_sided, 6), "\n")

alpha <- 0.05
if (p_one_sided < alpha) {
  cat("Conclusión: Se RECHAZA H0. La evidencia sugiere un incremento > 50 mil litros por cada 500 mil pesos.\n")
} else {
  cat("Conclusión: NO se rechaza H0 al 5%. No hay evidencia suficiente para afirmar un incremento > 50 mil litros por cada 500 mil pesos.\n")
}

# %%
p_line <- ggplot(data.frame(x = x, y = y), aes(x = x, y = y)) +
  geom_point(aes(color = "Datos")) +
  geom_smooth(method = "lm", se = FALSE, aes(color = "Recta ajustada")) +
  scale_color_manual(values = c("Datos" = "black", "Recta ajustada" = "red")) +
  labs(x = "Inversión en redes (millones de pesos)",
       y = "Ventas (millones de litros)",
       title = "RLS: Ventas ~ Inversión",
       color = "Elementos") +
  theme_minimal() +
  theme(panel.grid.minor = element_line(linetype = "dashed"),
        legend.position = "bottom")

line_path <- file.path(PLOTS_DIR, "shampoo_rls_line.png")
ggsave(line_path, p_line, dpi = 300, width = 6.5, height = 4.5)
cat("Figura guardada en:", line_path, "\n")
print(p_line)

# %%
