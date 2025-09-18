set.seed(42)
n <- 300
variable_independiente <- rnorm(n)
variable_dependiente  <- 5 + 2*rnorm(n) + rnorm(n)

df <- data.frame(variable_dependiente, variable_independiente)

# OLS
modelo  <- lm(variable_dependiente ~ variable_independiente, data = df)
residuos <- resid(modelo)

# Jarque–Bera
alpha <- 0.05
if (!requireNamespace("tseries", quietly = TRUE)) {
  install.packages("tseries")
}
library(tseries)
jb <- jarque.bera.test(residuos)

cat(sprintf("JB: %.4f\np-valor: %.6f\n", as.numeric(jb$statistic), jb$p.value))
if (jb$p.value < alpha) {
  cat("Rechazamos H0: los residuos no son normales.\n")
} else {
  cat("No rechazamos H0: no hay evidencia contra la normalidad.\n")
}
