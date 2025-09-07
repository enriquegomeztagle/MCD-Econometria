%% 

df = readtable('../../data/datos_fuerza_levantamiento.csv');
head(df)

%%
fprintf('Generando gráfico de dispersión...\n');
figure;
scatter(df.Fuerza_del_brazo_x, df.Levantamiento_dinamico_y);
xlabel('Fuerza del brazo, x');
ylabel('Levantamiento dinámico, y');
title('Dispersión: y vs. x');
grid on;
grid minor;


output_dir = '../../plots/matlab/ejercicio1';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
saveas(gcf, fullfile(output_dir, 'scatter_plot.png'), 'png');

%%
fprintf('Generando estadísticas descriptivas...\n');
desc = summary(df(:, {'Fuerza_del_brazo_x', 'Levantamiento_dinamico_y'}));
disp(desc);

%%
fprintf('Calculando coeficiente de correlación de Pearson...\n');
x = df.Fuerza_del_brazo_x;
y = df.Levantamiento_dinamico_y;

[r, pval] = corr(x, y);
fprintf('r (Pearson) = %.4f\n', r);
fprintf('p-valor = %.6f\n', pval);

fprintf('Realizando prueba de hipótesis...\n');
alpha = 0.05;
if pval < alpha
    fprintf('Conclusión: se rechaza H0; evidencia de relación lineal (α=0.05).\n');
else
    fprintf('Conclusión: no se rechaza H0; no hay evidencia suficiente de relación lineal (α=0.05).\n');
end

%%

X = [ones(length(x), 1), x];
[beta, ~, ~, ~, stats] = regress(y, X);
fprintf('Parámetros estimados:\n');
fprintf('Intercepto: %.4f\n', beta(1));
fprintf('Pendiente: %.4f\n', beta(2));

fprintf('\nResumen del modelo:\n');
fprintf('R^2: %.4f\n', stats(1));
fprintf('F-statistic: %.4f\n', stats(2));
fprintf('p-value: %.6f\n', stats(3));
fprintf('Error estándar: %.4f\n', sqrt(stats(4)));

%%
fprintf('Estimando modelo de regresión lineal simple por MCO...\n');
x0 = 30.0;


y_pred = beta(1) + beta(2) * x0;


n = length(x);
x_bar = mean(x);
Sxx = sum((x - x_bar).^2);
MSE = stats(4);


se_mean = sqrt(MSE * (1/n + (x0 - x_bar)^2 / Sxx));
t_crit = tinv(0.975, n-2);


ci_lower = y_pred - t_crit * se_mean;
ci_upper = y_pred + t_crit * se_mean;


se_pred = sqrt(MSE * (1 + 1/n + (x0 - x_bar)^2 / Sxx));
pi_lower = y_pred - t_crit * se_pred;
pi_upper = y_pred + t_crit * se_pred;

fprintf('Estimación puntual E[y|x=%.1f] = %.4f\n', x0, y_pred);
fprintf('Intervalo de confianza al 95%%:\n');
fprintf('[%.4f, %.4f]\n', ci_lower, ci_upper);

fprintf('\nIntervalo de predicción al 95%% (para una observación nueva):\n');
fprintf('[%.4f, %.4f]\n', pi_lower, pi_upper);

%%
resid = y - (beta(1) + beta(2) * x);
fprintf('Generando gráfico de residuales...\n');
figure;
scatter(x, resid);
hold on;
yline(0, '--');
xlabel('Fuerza del brazo, x');
ylabel('Residuales');
title('Residuales vs. x');
grid on;
grid minor;

saveas(gcf, '../../plots/matlab/ejercicio1/residuals_plot.png', 'png');

fprintf('Media de residuales (debe ser ~0): %.6f\n', mean(resid));
fprintf('Desviación estándar de residuales: %.6f\n', std(resid, 1));
corr_rx = corr(x, resid);
fprintf('Correlación(x, residuales) = %.6f (debe ser cercana a 0 en MCO con intercepto)\n', corr_rx);

%%
