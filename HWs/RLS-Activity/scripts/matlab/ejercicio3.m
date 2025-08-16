%%

DATA_CANDIDATES = {
    '../../data/flowers.xlsx',
    '../../data/flowers.csv'
};
PLOTS_DIR = '../../plots/matlab/ejercicio3';
if ~exist(PLOTS_DIR, 'dir')
    mkdir(PLOTS_DIR);
end


DATA_PATH = '';
for i = 1:length(DATA_CANDIDATES)
    if exist(DATA_CANDIDATES{i}, 'file')
        DATA_PATH = DATA_CANDIDATES{i};
        break;
    end
end

if isempty(DATA_PATH)
    error('No se encontró ningún archivo de datos. Esperaba ../data/flowers.xlsx o ../data/flowers.csv');
end

fprintf('Cargando: %s\n', DATA_PATH);
if endsWith(DATA_PATH, '.xlsx')
    df = readtable(DATA_PATH);
else
    df = readtable(DATA_PATH);
end

fprintf('\nPrimeras filas:\n');
disp(head(df));


x_candidates = {'flores', 'Flores', 'X', 'x'};
y_candidates = {'producci_n', 'producción', 'produccion', 'Producción', 'Y', 'y'};

x_col = '';
y_col = '';
for i = 1:length(x_candidates)
    if ismember(x_candidates{i}, df.Properties.VariableNames)
        x_col = x_candidates{i};
        break;
    end
end
for i = 1:length(y_candidates)
    if ismember(y_candidates{i}, df.Properties.VariableNames)
        y_col = y_candidates{i};
        break;
    end
end

% Verify that columns were found
if isempty(x_col) || isempty(y_col)
    fprintf('ERROR: No se pudieron detectar las columnas x e y\n');
    fprintf('Columnas disponibles: ');
    fprintf('%s ', df.Properties.VariableNames{:});
    fprintf('\n');
    fprintf('Nombres originales (si están disponibles): ');
    if isprop(df, 'VariableDescriptions')
        fprintf('%s ', df.Properties.VariableDescriptions{:});
    end
    fprintf('\n');
    return;
end

% Debug: show data types and first few values
fprintf('Tipo de datos de x: %s\n', class(df.(x_col)));
fprintf('Tipo de datos de y: %s\n', class(df.(y_col)));

% Try to convert data, handling different data types
if iscell(df.(x_col))
    x = str2double(df.(x_col));
else
    x = double(df.(x_col));
end

if iscell(df.(y_col))
    y = str2double(df.(y_col));
else
    y = double(df.(y_col));
end

% Check for NaN values and remove them
nan_mask = ~isnan(x) & ~isnan(y);
x = x(nan_mask);
y = y(nan_mask);

n = sum(nan_mask);
fprintf('\nColumnas detectadas -> x: %s, y: %s; n = %d (después de remover NaN)\n', x_col, y_col, n);

% Debug: show first few values
if n > 0
    fprintf('Primeros valores de x: ');
    fprintf('%.2f ', x(1:min(5,length(x))));
    fprintf('\n');
    fprintf('Primeros valores de y: ');
    fprintf('%.2f ', y(1:min(5,length(y))));
    fprintf('\n');
else
    fprintf('ERROR: No hay datos válidos después de la conversión!\n');
    fprintf('Valores originales de x (primeros 5): ');
    fprintf('%s ', string(df.(x_col)(1:min(5,height(df)))));
    fprintf('\n');
    fprintf('Valores originales de y (primeros 5): ');
    fprintf('%s ', string(df.(y_col)(1:min(5,height(df)))));
    fprintf('\n');
    return;
end

%%
fprintf('\n(a) Gráfico de dispersión y estadísticas descriptivas...\n');
figure;
scatter(x, y);
xlabel('Flores procesadas, x (miles)');
ylabel('Producción de esencia, y (onzas)');
title('Dispersión: Producción vs. Flores');
grid on;
grid minor;

scatter_path = fullfile(PLOTS_DIR, 'scatter_flowers.png');
saveas(gcf, scatter_path, 'png');
fprintf('Figura guardada en: %s\n', scatter_path);


% Calculate statistics for each variable separately
desc_stats = [mean(x), mean(y); 
              median(x), median(y); 
              std(x, 1), std(y, 1); 
              min(x), min(y); 
              quantile(x, 0.25), quantile(y, 0.25); 
              quantile(x, 0.75), quantile(y, 0.75); 
              max(x), max(y)];
desc_table = array2table(desc_stats, 'RowNames', {'media', 'mediana', 'desv_est', ...
                                                  'min', 'Q1', 'Q3', 'max'}, ...
                         'VariableNames', {'x_flores', 'y_produccion'});
fprintf('\nEstadísticas descriptivas (redondeadas):\n');
disp(round(desc_table, 3));

%%
fprintf('\n(b) Relación lineal: signo y evidencia estadística...\n');
[r, pval] = corr(x, y);
fprintf('Coef. de correlación de Pearson r = %.4f\n', r);
fprintf('p-valor (bilateral) = %.6f\n', pval);
if r > 0
    fprintf('Dirección: positiva\n');
elseif r < 0
    fprintf('Dirección: negativa\n');
else
    fprintf('Dirección: nula\n');
end

z = atanh(r);
se_z = 1/sqrt(n-3);
z_crit = norminv(0.975);
lo_r = tanh(z - z_crit*se_z);
hi_r = tanh(z + z_crit*se_z);
fprintf('IC 95%% para r: [%.4f, %.4f]\n', lo_r, hi_r);

%%
fprintf('\n(c) Ajuste RLS (MCO) y verificación de b0, b1, S^2...\n');

X = [ones(n, 1), x];
[beta, ~, ~, ~, stats] = regress(y, X);
b0_hat = beta(1);
b1_hat = beta(2);

resid = y - (beta(1) + beta(2) * x);
SSE = sum(resid.^2);
S2 = SSE/(n-2);

fprintf('b0_hat = %.4f\n', b0_hat);
fprintf('b1_hat = %.4f\n', b1_hat);
fprintf('S^2 (SSE/(n-2)) = %.4f\n', S2);
fprintf('\nResumen del modelo:\n');
fprintf('R^2: %.4f\n', stats(1));
fprintf('F-statistic: %.4f\n', stats(2));
fprintf('p-value: %.6f\n', stats(3));
fprintf('Error estándar: %.4f\n', sqrt(stats(4)));


figure;
scatter(x, y, 'DisplayName', 'Datos');
hold on;
xx = linspace(min(x), max(x), 100);
y_hat = beta(1) + beta(2) * xx;
plot(xx, y_hat, 'DisplayName', 'Recta ajustada');
xlabel('Flores procesadas, x (miles)');
ylabel('Producción de esencia, y (onzas)');
title('RLS: y ~ x');
legend('Location', 'best');
grid on;
grid minor;

line_path = fullfile(PLOTS_DIR, 'rls_line.png');
saveas(gcf, line_path, 'png');
fprintf('Figura guardada en: %s\n', line_path);

%%
fprintf('\n(c.1) Cálculos manuales para trazabilidad y verificación...\n');
Sxx = sum((x - mean(x)).^2);
Syy = sum((y - mean(y)).^2);
Sxy = sum((x - mean(x)).*(y - mean(y)));

b1_manual = Sxy / Sxx;
b0_manual = mean(y) - b1_manual * mean(x);
SSE_manual = sum((y - (b0_manual + b1_manual*x)).^2);
S2_manual = SSE_manual / (n - 2);

fprintf('Sxx = %.6f, Syy = %.6f, Sxy = %.6f\n', Sxx, Syy, Sxy);
fprintf('b0_manual = %.4f, b1_manual = %.4f\n', b0_manual, b1_manual);
fprintf('S^2_manual = %.4f\n', S2_manual);

b0_ref = 1.38; b1_ref = 0.52; S2_ref = 0.206;
fprintf('\nComparación con referencia (tolerancia +/- 0.03 para b0/b1 y +/- 0.02 para S^2):\n');
fprintf('|b0_manual - %.2f| = %.4f\n', b0_ref, abs(b0_manual - b0_ref));
fprintf('|b1_manual - %.2f| = %.4f\n', b1_ref, abs(b1_manual - b1_ref));
fprintf('|S^2_manual - %.3f| = %.4f\n', S2_ref, abs(S2_manual - S2_ref));
fprintf('Coincide b0? %s\n', mat2str(abs(b0_manual - b0_ref) <= 0.03));
fprintf('Coincide b1? %s\n', mat2str(abs(b1_manual - b1_ref) <= 0.03));
fprintf('Coincide S^2? %s\n', mat2str(abs(S2_manual - S2_ref) <= 0.02));

%%
fprintf('\n(d) ANOVA de la regresión y prueba F...\n');
y_bar = mean(y);
fitted_vals = beta(1) + beta(2) * x;
SSR = sum((fitted_vals - y_bar).^2);
SSE = sum((y - fitted_vals).^2);
SST = SSR + SSE;

DF_model = 1;
DF_resid = n - 2;
DF_total = n - 1;

MSR = SSR/DF_model;
MSE = SSE/DF_resid;
F_stat = MSR/MSE;
p_F = 1 - fcdf(F_stat, DF_model, DF_resid);

anova_table = array2table([SSR, SSE, SST; DF_model, DF_resid, DF_total; ...
                           MSR, MSE, NaN; F_stat, NaN, NaN; p_F, NaN, NaN]', ...
                          'VariableNames', {'SC', 'gl', 'CM', 'F', 'PR_F'}, ...
                          'RowNames', {'Regresión', 'Error', 'Total'});
fprintf('Tabla ANOVA (redondeada):\n');
disp(round(anova_table, 4));
fprintf('F = %.4f, df1 = %d, df2 = %d, p-valor = %.6f\n', F_stat, DF_model, DF_resid, p_F);

%%
fprintf('\n(d.1) Diagnósticos del modelo...\n');
resid = y - (beta(1) + beta(2) * x);
fitted = fitted_vals;


figure;
scatter(fitted, resid);
hold on;
yline(0, 'k--', 'LineWidth', 1);
xlabel('Valores ajustados');
ylabel('Residuales');
title('Residuales vs. Ajustados');
grid on;
grid minor;

resid_fit_path = fullfile(PLOTS_DIR, 'residuals_vs_fitted.png');
saveas(gcf, resid_fit_path, 'png');
fprintf('Figura guardada en: %s\n', resid_fit_path);


figure;
qqplot(resid);
title('QQ-plot de residuales');

qq_path = fullfile(PLOTS_DIR, 'qqplot_residuals.png');
saveas(gcf, qq_path, 'png');
fprintf('Figura guardada en: %s\n', qq_path);


figure;
histogram(resid, 8, 'EdgeColor', 'black');
title('Histograma de residuales');
xlabel('Residual');
ylabel('Frecuencia');

hist_path = fullfile(PLOTS_DIR, 'hist_residuals.png');
saveas(gcf, hist_path, 'png');
fprintf('Figura guardada en: %s\n', hist_path);



skew = skewness(resid);
kurt = kurtosis(resid);
jb_stat = n * (skew^2/6 + (kurt-3)^2/24);
jb_p = 1 - chi2cdf(jb_stat, 2);
fprintf('Jarque-Bera: JB = %.4f, p = %.6f, skew = %.4f, kurt = %.4f\n', ...
        jb_stat, jb_p, skew, kurt);

%%
fprintf('\n(e) Error estándar de la pendiente e IC al 95%%...\n');
Sxx = sum((x - mean(x)).^2);
se_b1 = sqrt(MSE / Sxx);
t_crit = tinv(0.975, DF_resid);
ci_b1 = [b1_hat - t_crit*se_b1, b1_hat + t_crit*se_b1];
fprintf('SE(b1) = %.6f\n', se_b1);
fprintf('IC 95%% para b1: [%.4f, %.4f]\n', ci_b1(1), ci_b1(2));

%%
fprintf('\n(f) Porcentaje de variabilidad explicada...\n');
R2 = SSR / SST;
fprintf('R^2 = %.4f -> %.2f%%\n', R2, R2*100);

%%
fprintf('\n(g) IC 95%% para E[y|x0]...\n');
x0 = 1.25;
pred_mean = beta(1) + beta(2) * x0;
se_mean = sqrt(MSE * (1/n + (x0 - mean(x))^2 / Sxx));
mean_hat = pred_mean;
ci_mean = [mean_hat - t_crit*se_mean, mean_hat + t_crit*se_mean];
fprintf('E[y|x0] puntual = %.4f\n', mean_hat);
fprintf('IC 95%% para E[y|x0]: [%.4f, %.4f]\n', ci_mean(1), ci_mean(2));

%%
fprintf('\n(h) Intervalo de predicción al 95%%...\n');
x0 = 1.95;
pred_obs = beta(1) + beta(2) * x0;
se_pred = sqrt(MSE * (1 + 1/n + (x0 - mean(x))^2 / Sxx));
mean_hat = pred_obs;
pi = [mean_hat - t_crit*se_pred, mean_hat + t_crit*se_pred];
fprintf('y_hat puntual = %.4f\n', mean_hat);
fprintf('PI 95%% para y|x0: [%.4f, %.4f]\n', pi(1), pi(2));

%%
