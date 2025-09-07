# %%
library(readr)
library(ggplot2)
library(dplyr)
library(broom)

df <- read_csv("../../data/datos_fuerza_levantamiento.csv")
head(df)

# %%
cat("Generando gráfico de dispersión...\n")

output_dir <- "../../plots/r/ejercicio1"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

p <- ggplot(df, aes(x = Fuerza_del_brazo_x, y = Levantamiento_dinamico_y)) +
  geom_point() +
  labs(x = "Fuerza del brazo, x", 
       y = "Levantamiento dinámico, y",
       title = "Dispersión: y vs. x") +
  theme_minimal() +
  theme(panel.grid.minor = element_line(linetype = "dashed"))

ggsave(file.path(output_dir, "scatter_plot.png"), p, dpi = 300, width = 6, height = 4)
print(p)

# %%
cat("Generando estadísticas descriptivas...\n")
desc <- df %>%
  select(Fuerza_del_brazo_x, Levantamiento_dinamico_y) %>%
  summary()
print(desc)

# %%
cat("Calculando coeficiente de correlación de Pearson...\n")
x <- df$Fuerza_del_brazo_x
y <- df$Levantamiento_dinamico_y

cor_result <- cor.test(x, y, method = "pearson")
r <- cor_result$estimate
pval <- cor_result$p.value

cat("r (Pearson) =", round(r, 4), "\n")
cat("p-valor =", round(pval, 6), "\n")

cat("Realizando prueba de hipótesis...\n")
alpha <- 0.05
if (pval < alpha) {
  cat("Conclusión: se rechaza H0; evidencia de relación lineal (α=0.05).\n")
} else {
  cat("Conclusión: no se rechaza H0; no hay evidencia suficiente de relación lineal (α=0.05).\n")
}

# %%
model <- lm(Levantamiento_dinamico_y ~ Fuerza_del_brazo_x, data = df)
cat("Parámetros estimados:\n")
print(coef(model))

cat("\nResumen del modelo:\n")
print(summary(model))

# %%
cat("Estimando modelo de regresión lineal simple por MCO...\n")
x0 <- 30.0

new_data <- data.frame(Fuerza_del_brazo_x = x0)
pred <- predict(model, new_data, interval = "confidence", level = 0.95)
pred_obs <- predict(model, new_data, interval = "prediction", level = 0.95)

cat("Estimación puntual E[y|x=", x0, "] =", round(pred[1], 4), "\n")
cat("Intervalo de confianza al 95% para la media condicional:\n")
print(pred)

cat("\nIntervalo de predicción al 95% (para una observación nueva):\n")
print(pred_obs)

# %%
resid <- residuals(model)
cat("Generando gráfico de residuales...\n")

p_resid <- ggplot(data.frame(x = x, resid = resid), aes(x = x, y = resid)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Fuerza del brazo, x", 
       y = "Residuales",
       title = "Residuales vs. x") +
  theme_minimal() +
  theme(panel.grid.minor = element_line(linetype = "dashed"))

ggsave(file.path(output_dir, "residuals_plot.png"), p_resid, dpi = 300, width = 6, height = 4)
print(p_resid)

cat("Media de residuales (debe ser ~0):", round(mean(resid), 6), "\n")
cat("Desviación estándar de residuales:", round(sd(resid), 6), "\n")
corr_rx <- cor(x, resid)
cat("Correlación(x, residuales) =", round(corr_rx, 6), "(debe ser cercana a 0 en MCO con intercepto)\n")

# %%
