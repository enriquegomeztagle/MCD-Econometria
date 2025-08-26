data = readtable('../../data/avocado_exports.csv', 'VariableNamingRule', 'preserve');
df.Ano = double(data.("Año"));
df.Bimestre = double(data.Bimestre);
df.t = double(data.t);
df.Toneladas = double(data.Toneladas);

media = mean(df.Toneladas);
desv = std(df.Toneladas);
cv = (desv / media) * 100;
fprintf('[1] Coeficiente de variación: %.2f%%\n', cv);

fprintf('\n[2] Estadísticas descriptivas (Toneladas):\n');
fprintf('Media: %.2f\n', mean(df.Toneladas));
fprintf('Desv. Est.: %.2f\n', std(df.Toneladas));
fprintf('Mínimo: %.2f\n', min(df.Toneladas));
fprintf('Máximo: %.2f\n', max(df.Toneladas));
fprintf('Q1: %.2f\n', prctile(df.Toneladas, 25));
fprintf('Q2: %.2f\n', prctile(df.Toneladas, 50));
fprintf('Q3: %.2f\n', prctile(df.Toneladas, 75));

figure;
plot(df.t, df.Toneladas, 'o-');
title('Serie bimestral de exportaciones de aguacate');
xlabel('t (bimestres desde 2019-1)');
ylabel('Toneladas');
grid on;
saveas(gcf, '../../plots/matlab/avocado_exports_bimestral.png');
close;

figure;
histogram(df.Toneladas, 10);
title('Distribución de exportaciones (Toneladas)');
xlabel('Toneladas');
ylabel('Frecuencia');
grid on;
saveas(gcf, '../../plots/matlab/avocado_exports_histogram.png');
close;

figure;
boxplot(df.Toneladas, 'Labels', {'Toneladas'});
title('Boxplot de exportaciones bimestrales (Toneladas)');
ylabel('Toneladas');
grid on;
saveas(gcf, '../../plots/matlab/avocado_exports_boxplot.png');
close;

X = [ones(length(df.t), 1), df.t];
y = df.Toneladas;
[beta_orig, ~, ~, ~, stats_orig] = regress(y, X);

pendiente = beta_orig(2) / 1000.0;
fprintf('\n[3] Coeficiente de pendiente muestral (miles de toneladas por bimestre): %.4f\n', pendiente);

fprintf('\n[4] Prueba de hipótesis para pendiente > 0 (serie original)\n');
fprintf('R^2: %.4f\n', stats_orig(1));
fprintf('F-statistic: %.4f\n', stats_orig(2));
fprintf('p-value: %.4f\n', stats_orig(3));

n = length(y);
p = 2; % number of parameters
dof = n - p;
residuals = y - X * beta_orig;
mse = sum(residuals.^2) / dof;
se_beta = sqrt(mse * inv(X' * X));
t_stat = beta_orig(2) / se_beta(2, 2);
p_two_sided = 2 * (1 - tcdf(abs(t_stat), dof));
p_one_sided = p_two_sided / 2;
if t_stat < 0
    p_one_sided = 1 - p_one_sided;
end

fprintf('t = %.3f, p una-cola = %.4g -> %s\n', t_stat, p_one_sided, ...
    ternary(p_one_sided < 0.05, 'Rechazamos H0: tendencia positiva', 'No se rechaza H0'));

k = 6;
promedio_global = mean(df.Toneladas);
indices_estacionales = zeros(k, 1);
for bim = 1:k
    indices_estacionales(bim) = mean(df.Toneladas(df.Bimestre == bim)) / promedio_global;
end

fprintf('\n[5] Índices estacionales (multiplicativos):\n');
for bim = 1:k
    fprintf('Bimestre %d: %.4f\n', bim, indices_estacionales(bim));
end
fprintf('Interpretación: valores >1 indican bimestres por encima del promedio; <1 por debajo.\n');

df.IndiceEstacional = indices_estacionales(df.Bimestre);
df.Deseasonalizada = df.Toneladas ./ df.IndiceEstacional;

figure;
plot(df.t, df.Deseasonalizada, 'o-');
title('Serie desestacionalizada (multiplicativa)');
xlabel('t');
ylabel('Toneladas desestacionalizadas');
grid on;
saveas(gcf, '../../plots/matlab/avocado_exports_deseasonalized.png');
close;

X_d = [ones(length(df.t), 1), df.t];
y_d = df.Deseasonalizada;
[beta_des, ~, ~, ~, stats_des] = regress(y_d, X_d);

fprintf('\n[6] Regresión con datos desestacionalizados:\n');
fprintf('R^2: %.4f\n', stats_des(1));
fprintf('F-statistic: %.4f\n', stats_des(2));
fprintf('p-value: %.4f\n', stats_des(3));

residuals_d = y_d - X_d * beta_des;
mse_d = sum(residuals_d.^2) / dof;
se_beta_d = sqrt(mse_d * inv(X_d' * X_d));
t_stat_d = beta_des(2) / se_beta_d(2, 2);
p_two_sided_d = 2 * (1 - tcdf(abs(t_stat_d), dof));
p_one_sided_d = p_two_sided_d / 2;
if t_stat_d < 0
    p_one_sided_d = 1 - p_one_sided_d;
end

fprintf('t = %.3f, p una-cola = %.4g -> %s\n', t_stat_d, p_one_sided_d, ...
    ternary(p_one_sided_d < 0.05, 'Rechazamos H0: tendencia positiva', 'No se rechaza H0'));

t_future = 32;
X_future = [1, t_future];
pred_mean = X_future * beta_orig;

t_critical = tinv(0.975, dof);
se_pred = sqrt(mse * (X_future * inv(X' * X) * X_future'));
ci_low = pred_mean - t_critical * se_pred;
ci_high = pred_mean + t_critical * se_pred;

fprintf('\n[7] IC 95%% para la tendencia (media esperada) en 2024-bim2 (t=32): [%.2f, %.2f] toneladas. Estimado: %.2f\n', ...
    ci_low, ci_high, pred_mean);

r2_orig = stats_orig(1);
r2_des = stats_des(1);
fprintf('\n[8] R^2 serie original: %.4f\n', r2_orig);
fprintf('[8] R^2 serie desestacionalizada: %.4f\n', r2_des);
fprintf('Explicación: al quitar la variación estacional, el componente sistemático por tiempo puede capturar mejor la tendencia subyacente (o a veces menos, si la estacionalidad ya explicaba variación alineada con t). Un cambio en R^2 refleja cuánto peso de la variabilidad se atribuye a la estacionalidad vs. la tendencia.\n');

fut.t = (31:36)';
fut.Ano = zeros(length(fut.t), 1);
fut.Bimestre = zeros(length(fut.t), 1);

function [year, bim] = t_to_year_bim(t)
    base_year = 2019;
    idx = t - 1;
    year = base_year + floor(idx / 6);
    bim = mod(idx, 6) + 1;
end

for i = 1:length(fut.t)
    [fut.Ano(i), fut.Bimestre(i)] = t_to_year_bim(fut.t(i));
end

X_fut = [ones(length(fut.t), 1), fut.t];
pred_orig = X_fut * beta_orig;
fut.Pronostico_Original = pred_orig;

pred_des = X_fut * beta_des;
fut.IndiceEstacional = indices_estacionales(fut.Bimestre);
fut.Pronostico_Deseason = pred_des .* fut.IndiceEstacional;

fprintf('\n[9] Pronósticos 2024 (toneladas):\n');
fprintf('Año\tBimestre\tt\tPronóstico Original\tPronóstico Desestacionalizado\n');
for i = 1:length(fut.t)
    fprintf('%d\t%d\t\t%d\t%.2f\t\t\t%.2f\n', ...
        fut.Ano(i), fut.Bimestre(i), fut.t(i), ...
        fut.Pronostico_Original(i), fut.Pronostico_Deseason(i));
end

figure;
plot(df.t, df.Toneladas, 'o-', 'DisplayName', 'Observado');
hold on;
plot(fut.t, fut.Pronostico_Original, 'o--', 'DisplayName', 'Pronóstico (original)');
title('Pronóstico con modelo original');
xlabel('t');
ylabel('Toneladas');
legend('Location', 'best');
grid on;
saveas(gcf, '../../plots/matlab/avocado_exports_original_forecast.png');
close;

figure;
plot(df.t, df.Toneladas, 'o-', 'DisplayName', 'Observado');
hold on;
plot(fut.t, fut.Pronostico_Deseason, 'o--', 'DisplayName', 'Pronóstico (desestacionalizado)');
title('Pronóstico con modelo desestacionalizado (reestacionalizado)');
xlabel('t');
ylabel('Toneladas');
legend('Location', 'best');
grid on;
saveas(gcf, '../../plots/matlab/avocado_exports_deseasonalized_forecast.png');
close;

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
