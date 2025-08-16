%%

DATA_PATH = '../../data/shampoo.csv';
PLOTS_DIR = '../../plots/matlab/ejercicio2';
if ~exist(PLOTS_DIR, 'dir')
    mkdir(PLOTS_DIR);
end

fprintf('Cargando: %s\n', DATA_PATH);
df = readtable(DATA_PATH);
fprintf('\nPrimeras filas:\n');
disp(head(df));


ventas_col_candidates = {'Ventas_Millones_lts', 'Ventas', 'ventas'};
inversion_col_candidates = {'Inversion_Millones_pesos', 'Inversion', 'inversion'};

ventas_col = '';
inversion_col = '';
for i = 1:length(ventas_col_candidates)
    if ismember(ventas_col_candidates{i}, df.Properties.VariableNames)
        ventas_col = ventas_col_candidates{i};
        break;
    end
end
for i = 1:length(inversion_col_candidates)
    if ismember(inversion_col_candidates{i}, df.Properties.VariableNames)
        inversion_col = inversion_col_candidates{i};
        break;
    end
end

% Debug: show data types and first few values
fprintf('Tipo de datos de x: %s\n', class(df.(inversion_col)));
fprintf('Tipo de datos de y: %s\n', class(df.(ventas_col)));

% Try to convert data, handling different data types
if iscell(df.(inversion_col))
    x = str2double(df.(inversion_col));
else
    x = double(df.(inversion_col));
end

if iscell(df.(ventas_col))
    y = str2double(df.(ventas_col));
else
    y = double(df.(ventas_col));
end

% Check for NaN values and remove them
nan_mask = ~isnan(x) & ~isnan(y);
x = x(nan_mask);
y = y(nan_mask);

n = sum(nan_mask);
fprintf('\nColumnas detectadas -> y: %s, x: %s; n = %d (después de remover NaN)\n', ventas_col, inversion_col, n);

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
    fprintf('%s ', string(df.(inversion_col)(1:min(5,height(df)))));
    fprintf('\n');
    fprintf('Valores originales de y (primeros 5): ');
    fprintf('%s ', string(df.(ventas_col)(1:min(5,height(df)))));
    fprintf('\n');
    return;
end

%%
fprintf('\n(a) Generando diagrama de dispersión y estadísticas descriptivas...\n');
figure;
scatter(x, y);
xlabel('Inversión en redes (millones de pesos)');
ylabel('Ventas (millones de litros)');
title('Dispersión: Ventas vs. Inversión');
grid on;
grid minor;

scatter_path = fullfile(PLOTS_DIR, 'scatter_plot.png');
saveas(gcf, scatter_path, 'png');
fprintf('Figura guardada en: %s\n', scatter_path);


% Calculate statistics for each variable separately
desc_stats = [mean(y), mean(x); 
              median(y), median(x); 
              std(y, 1), std(x, 1); 
              min(y), min(x); 
              quantile(y, 0.25), quantile(x, 0.25); 
              quantile(y, 0.75), quantile(x, 0.75); 
              max(y), max(x)];
desc_table = array2table(desc_stats, 'RowNames', {'media', 'mediana', 'desv_est', ...
                                                  'min', 'Q1', 'Q3', 'max'}, ...
                         'VariableNames', {'Ventas_y', 'Inversion_x'});
fprintf('\nEstadísticas descriptivas:\n');
disp(desc_table);

%%
fprintf('\n(b) Correlación de Pearson e intervalo de confianza al 95%%...\n');
[r, pval] = corr(x, y);
fprintf('r = %.4f\n', r);
fprintf('p-valor (bilateral) = %.6f\n', pval);


z = atanh(r);  
se_z = 1 / sqrt(n - 3);
z_crit = norminv(0.975);  % 1.96
lo_z = z - z_crit * se_z;
hi_z = z + z_crit * se_z;
lo_r = tanh(lo_z);
hi_r = tanh(hi_z);
fprintf('IC 95%%: [%.4f, %.4f]\n', lo_r, hi_r);

%%
fprintf('\n(c) Ajustando RLS (MCO): y = beta0 + beta1*x + e ...\n');

X = [ones(n, 1), x];
[beta, ~, ~, ~, stats] = regress(y, X);
fprintf('\nParámetros estimados:\n');
fprintf('Intercepto: %.4f\n', beta(1));
fprintf('Pendiente: %.4f\n', beta(2));
fprintf('\nResumen del modelo:\n');
fprintf('R^2: %.4f\n', stats(1));
fprintf('F-statistic: %.4f\n', stats(2));
fprintf('p-value: %.6f\n', stats(3));
fprintf('Error estándar: %.4f\n', sqrt(stats(4)));

%%
fprintf('\n(d) Prueba de hipótesis (one-sided, alpha=0.05)...\n');

beta1_hat = beta(2);
se_beta1 = sqrt(stats(4) / sum((x - mean(x)).^2));
df_resid = n - 2;

beta1_H0 = 0.1;

t_stat = (beta1_hat - beta1_H0) / se_beta1;
p_one_sided = 1 - tcdf(t_stat, df_resid);  

fprintf('beta1_hat = %.6f\n', beta1_hat);
fprintf('SE(beta1) = %.6f\n', se_beta1);
fprintf('t = %.4f, df = %d\n', t_stat, df_resid);
fprintf('p-valor (H1: beta1 > 0.1) = %.6f\n', p_one_sided);

alpha = 0.05;
if p_one_sided < alpha
    fprintf('Conclusión: Se RECHAZA H0. La evidencia sugiere un incremento > 50 mil litros por cada 500 mil pesos.\n');
else
    fprintf('Conclusión: NO se rechaza H0 al 5%%. No hay evidencia suficiente para afirmar un incremento > 50 mil litros por cada 500 mil pesos.\n');
end

%%
figure;
scatter(x, y, 'DisplayName', 'Datos');
hold on;
xx = linspace(min(x), max(x), 100);
y_hat = beta(1) + beta(2) * xx;
plot(xx, y_hat, 'DisplayName', 'Recta ajustada');
xlabel('Inversión en redes (millones de pesos)');
ylabel('Ventas (millones de litros)');
title('RLS: Ventas ~ Inversión');
legend('Location', 'best');
grid on;
grid minor;

line_path = fullfile(PLOTS_DIR, 'shampoo_rls_line.png');
saveas(gcf, line_path, 'png');
fprintf('Figura guardada en: %s\n', line_path);

%%
