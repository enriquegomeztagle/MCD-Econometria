library(ggplot2)
library(dplyr)
library(gridExtra)

df <- read.csv("../../data/avocado_exports.csv")
df$Año <- as.integer(df$Año)
df$Bimestre <- as.integer(df$Bimestre)
df$t <- as.integer(df$t)
df$Toneladas <- as.numeric(df$Toneladas)

media <- mean(df$Toneladas)
desv <- sd(df$Toneladas)
cv <- (desv / media) * 100
cat(sprintf("[1] Coeficiente de variación: %.2f%%\n", cv))

cat("\n[2] Estadísticas descriptivas (Toneladas):\n")
print(summary(df$Toneladas))

p1 <- ggplot(df, aes(x = t, y = Toneladas)) +
  geom_line() +
  geom_point() +
  labs(title = "Serie bimestral de exportaciones de aguacate",
       x = "t (bimestres desde 2019-1)",
       y = "Toneladas") +
  theme_minimal() +
  theme(panel.grid = element_line(color = "gray90"))

ggsave("../../plots/r/avocado_exports_bimestral.png", p1, width = 8, height = 6)
print(p1)

p2 <- ggplot(df, aes(x = Toneladas)) +
  geom_histogram(bins = 10, fill = "steelblue", alpha = 0.7) +
  labs(title = "Distribución de exportaciones (Toneladas)",
       x = "Toneladas",
       y = "Frecuencia") +
  theme_minimal() +
  theme(panel.grid = element_line(color = "gray90"))

ggsave("../../plots/r/avocado_exports_histogram.png", p2, width = 8, height = 6)
print(p2)

p3 <- ggplot(df, aes(y = Toneladas)) +
  geom_boxplot(fill = "steelblue", alpha = 0.7) +
  labs(title = "Boxplot de exportaciones bimestrales (Toneladas)",
       y = "Toneladas") +
  theme_minimal() +
  theme(panel.grid = element_line(color = "gray90"))

ggsave("../../plots/r/avocado_exports_boxplot.png", p3, width = 8, height = 6)
print(p3)

model_orig <- lm(Toneladas ~ t, data = df)

pendiente <- coef(model_orig)["t"] / 1000.0
cat(sprintf("\n[3] Coeficiente de pendiente muestral (miles de toneladas por bimestre): %.4f\n", pendiente))

cat("\n[4] Prueba de hipótesis para pendiente > 0 (serie original)\n")
print(summary(model_orig))

t_stat <- summary(model_orig)$coefficients["t", "t value"]
p_two_sided <- summary(model_orig)$coefficients["t", "Pr(>|t|)"]
p_one_sided <- p_two_sided / 2
if (t_stat < 0) {
  p_one_sided <- 1 - p_one_sided
}

cat(sprintf("t = %.3f, p una-cola = %.4g -> %s\n", t_stat, p_one_sided,
            ifelse(p_one_sided < 0.05, "Rechazamos H0: tendencia positiva", "No se rechaza H0")))

k <- 6
promedio_global <- mean(df$Toneladas)
indices_estacionales <- numeric(k)
for (bim in 1:k) {
  indices_estacionales[bim] <- mean(df$Toneladas[df$Bimestre == bim]) / promedio_global
}

cat("\n[5] Índices estacionales (multiplicativos):\n")
for (bim in 1:k) {
  cat(sprintf("Bimestre %d: %.4f\n", bim, indices_estacionales[bim]))
}
cat("Interpretación: valores >1 indican bimestres por encima del promedio; <1 por debajo.\n")

df$IndiceEstacional <- indices_estacionales[df$Bimestre]
df$Deseasonalizada <- df$Toneladas / df$IndiceEstacional

p4 <- ggplot(df, aes(x = t, y = Deseasonalizada)) +
  geom_line() +
  geom_point() +
  labs(title = "Serie desestacionalizada (multiplicativa)",
       x = "t",
       y = "Toneladas desestacionalizadas") +
  theme_minimal() +
  theme(panel.grid = element_line(color = "gray90"))

ggsave("../../plots/r/avocado_exports_deseasonalized.png", p4, width = 8, height = 6)
print(p4)

model_des <- lm(Deseasonalizada ~ t, data = df)

cat("\n[6] Regresión con datos desestacionalizados:\n")
print(summary(model_des))

t_stat_d <- summary(model_des)$coefficients["t", "t value"]
p_two_sided_d <- summary(model_des)$coefficients["t", "Pr(>|t|)"]
p_one_sided_d <- p_two_sided_d / 2
if (t_stat_d < 0) {
  p_one_sided_d <- 1 - p_one_sided_d
}

cat(sprintf("t = %.3f, p una-cola = %.4g -> %s\n", t_stat_d, p_one_sided_d,
            ifelse(p_one_sided_d < 0.05, "Rechazamos H0: tendencia positiva", "No se rechaza H0")))

t_future <- 32
X_future <- data.frame(t = t_future)
pred_mean <- predict(model_orig, newdata = X_future, interval = "confidence", level = 0.95)
ci_low <- pred_mean[1, "lwr"]
ci_high <- pred_mean[1, "upr"]
mean_pred <- pred_mean[1, "fit"]

cat(sprintf("\n[7] IC 95%% para la tendencia (media esperada) en 2024-bim2 (t=32): [%.2f, %.2f] toneladas. Estimado: %.2f\n",
            ci_low, ci_high, mean_pred))

pi <- predict(model_orig, newdata = X_future, interval = "prediction", level = 0.95)
cat(sprintf("[7-extra] IC 95%% de predicción (valor observado esperado) en 2024-bim2: [%.2f, %.2f]\n",
            pi[1, "lwr"], pi[1, "upr"]))

r2_orig <- summary(model_orig)$r.squared
r2_des <- summary(model_des)$r.squared
cat(sprintf("\n[8] R^2 serie original: %.4f\n", r2_orig))
cat(sprintf("[8] R^2 serie desestacionalizada: %.4f\n", r2_des))
cat("Explicación: al quitar la variación estacional, el componente sistemático por tiempo puede capturar mejor la tendencia subyacente (o a veces menos, si la estacionalidad ya explicaba variación alineada con t). Un cambio en R^2 refleja cuánto peso de la variabilidad se atribuye a la estacionalidad vs. la tendencia.\n")

fut <- data.frame(t = 31:36)

t_to_year_bim <- function(t) {
  base_year <- 2019
  idx <- t - 1
  year <- base_year + floor(idx / 6)
  bim <- (idx %% 6) + 1
  return(c(year, bim))
}

yb <- t(sapply(fut$t, t_to_year_bim))
fut$Año <- yb[, 1]
fut$Bimestre <- yb[, 2]

pred_orig <- predict(model_orig, newdata = fut)
fut$Pronostico_Original <- pred_orig

pred_des <- predict(model_des, newdata = fut)
fut$IndiceEstacional <- indices_estacionales[fut$Bimestre]
fut$Pronostico_Deseason <- pred_des * fut$IndiceEstacional

cat("\n[9] Pronósticos 2024 (toneladas):\n")
print(round(fut[, c("Año", "Bimestre", "t", "Pronostico_Original", "Pronostico_Deseason")], 2))

pred_orig_full <- predict(model_orig, newdata = fut, interval = "confidence", level = 0.95)
fut$IC_orig_low <- pred_orig_full[, "lwr"]
fut$IC_orig_high <- pred_orig_full[, "upr"]

pred_des_full <- predict(model_des, newdata = fut, interval = "prediction", level = 0.95)
fut$IC_des_low <- pred_des_full[, "lwr"] * fut$IndiceEstacional
fut$IC_des_high <- pred_des_full[, "upr"] * fut$IndiceEstacional

cat("\n[9-extra] Pronósticos con intervalos de confianza:\n")
print(round(fut[, c("Año", "Bimestre", "Pronostico_Original", "IC_orig_low", "IC_orig_high",
                     "Pronostico_Deseason", "IC_des_low", "IC_des_high")], 2))

p5 <- ggplot() +
  geom_line(data = df, aes(x = t, y = Toneladas), color = "black") +
  geom_point(data = df, aes(x = t, y = Toneladas), color = "black") +
  geom_line(data = fut, aes(x = t, y = Pronostico_Original), color = "red", linetype = "dashed") +
  geom_point(data = fut, aes(x = t, y = Pronostico_Original), color = "red") +
  labs(title = "Pronóstico con modelo original",
       x = "t",
       y = "Toneladas") +
  theme_minimal() +
  theme(panel.grid = element_line(color = "gray90"))

ggsave("../../plots/r/avocado_exports_original_forecast.png", p5, width = 8, height = 6)
print(p5)

p6 <- ggplot() +
  geom_line(data = df, aes(x = t, y = Toneladas), color = "black") +
  geom_point(data = df, aes(x = t, y = Toneladas), color = "black") +
  geom_line(data = fut, aes(x = t, y = Pronostico_Deseason), color = "blue", linetype = "dashed") +
  geom_point(data = fut, aes(x = t, y = Pronostico_Deseason), color = "blue") +
  labs(title = "Pronóstico con modelo desestacionalizado (reestacionalizado)",
       x = "t",
       y = "Toneladas") +
  theme_minimal() +
  theme(panel.grid = element_line(color = "gray90"))

ggsave("../../plots/r/avocado_exports_deseasonalized_forecast.png", p6, width = 8, height = 6)
print(p6)
