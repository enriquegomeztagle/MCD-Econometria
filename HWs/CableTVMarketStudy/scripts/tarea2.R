install.packages(c("readxl", "ggplot2", "dplyr"))

library(readxl)
library(ggplot2)
library(dplyr)
library(scales)

tryCatch({
  df <- read_excel("../data/cableTV.xlsx")
  cat("File read successfully\n")
  read_success <- TRUE
}, error = function(e) {
  cat("Error while reading the file:", e$message, "\n")
  read_success <<- FALSE
})

if (read_success) {
  cat("--------------------------------\n")
  cat("Dataframe head:\n")
  print(head(df, 10))
}

if (read_success) {
  cat("--------------------------------\n")
  cat("Dataframe info:\n")
  cat("Shape:", nrow(df), "x", ncol(df), "\n")
  cat("Column names:", paste(names(df), collapse = ", "), "\n")
  str(df)
}

if (read_success) {
  cat("--------------------------------\n")
  cat("Dataframe describe:\n")
  print(summary(df))
}

if (read_success) {
  cat("--------------------------------\n")
  cat("Dataframe shape:\n")
  cat(nrow(df), "x", ncol(df), "\n")
}

if (read_success) {
  cat("--------------------------------\n")
  cat("Dataframe missing values:\n")
  missing_counts <- sapply(df, function(x) sum(is.na(x)))
  print(missing_counts)
  
  cat("--------------------------------\n")
  cat("Dataframe unique values:\n")
  unique_counts <- sapply(df, function(x) length(unique(x[!is.na(x)])))
  print(unique_counts)
}

if (read_success) {
  quant_vars <- c("adultos", "ninos", "teles", "tvtot", "renta", "valor")
  
  if (!dir.exists("../plots/r")) {
    dir.create("../plots/r", recursive = TRUE)
  }
  
  for (var in quant_vars) {
    if (var %in% names(df)) {
      values <- df[[var]]
      label <- var
      
      if (var == "valor") {
        values <- values / 1000
        label <- paste(label, "(thousands of pesos)")
      }
      
      p <- ggplot(data.frame(values = values), aes(x = values)) +
        geom_histogram(bins = 30, color = "black", fill = "lightblue", alpha = 0.7) +
        labs(x = label, y = "Frequency", title = paste("Frequency for", label)) +
        theme_minimal() +
        theme(panel.grid = element_line(colour = alpha("grey", 0.5), linetype = "dashed"))
      
      ggsave(paste0("../plots/r/frequency_", var, ".png"), plot = p, width = 8, height = 6, dpi = 300)
    }
  }
}

if (read_success) {
  cat_vars <- c("colonia", "tipo")
  
  for (var in cat_vars) {
    if (var %in% names(df)) {
      cat("=== Frequency of '", var, "' ===\n", sep = "")
      freq_table <- table(df[[var]], useNA = "ifany")
      freq_sorted <- sort(freq_table, decreasing = TRUE)
      print(freq_sorted)
      cat("\n")
    }
  }
}
