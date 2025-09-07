%%

DATA_CANDIDATES = {
    '../../data/cableTV.xlsx',
    '../../data/cableTV.csv'
};
PLOTS_DIR = '../../plots/matlab/ejercicio4';
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
    error('No se encontró cableTV.{xlsx,csv} en ../data/');
end

fprintf('Cargando: %s\n', DATA_PATH);
if endsWith(DATA_PATH, '.xlsx')
    df = readtable(DATA_PATH);
else
    df = readtable(DATA_PATH);
end

fprintf('\nColumnas disponibles: ');
disp(df.Properties.VariableNames);


df.Properties.VariableNames = lower(df.Properties.VariableNames);

expected = {'obs', 'colonia', 'manzana', 'adultos', 'ninos', 'teles', 'renta', 'tvtot', 'tipo', 'valor'};
missing_cols = setdiff(expected, df.Properties.VariableNames);
if ~isempty(missing_cols)
    fprintf('Advertencia: faltan columnas esperadas: ');
    disp(missing_cols);
end

df.x_valor_miles = df.valor / 1000.0;
x_name = 'x_valor_miles';
y_name = 'renta';
fprintf('\nPrimeras filas:\n');
disp(head(df(:, {'obs', 'colonia', 'manzana', 'adultos', 'ninos', 'teles', y_name, 'tvtot', 'tipo', 'valor', x_name})));

function model = fit_ols_xy(x, y)
    n = length(x);
    X = [ones(n, 1), x];
    [beta, ~, ~, ~, stats] = regress(y, X);
    model.beta = beta;
    model.stats = stats;
    model.resid = y - (beta(1) + beta(2) * x);
    model.df_resid = n - 2;
    model.ess = stats(1) * sum((y - mean(y)).^2);  
    model.ssr = sum(model.resid.^2);  
    model.df_model = 1;
    model.rsquared = stats(1);
end

%%
fprintf('\n(a) Ajuste MCO con todos los datos y gráficas...\n');
mask_full = true(height(df), 1);

% Debug: show data types and first few values
fprintf('Tipo de datos de x: %s\n', class(df{mask_full, x_name}));
fprintf('Tipo de datos de y: %s\n', class(df{mask_full, y_name}));

% Try to convert data, handling different data types
if iscell(df{mask_full, x_name})
    x = str2double(df{mask_full, x_name});
else
    x = double(df{mask_full, x_name});
end

if iscell(df{mask_full, y_name})
    y = str2double(df{mask_full, y_name});
else
    y = double(df{mask_full, y_name});
end

% Check for NaN values and remove them
nan_mask = ~isnan(x) & ~isnan(y);
x = x(nan_mask);
y = y(nan_mask);

n_full = sum(nan_mask);
fprintf('Datos válidos después de remover NaN: %d\n', n_full);

% Debug: show first few values
if n_full > 0
    fprintf('Primeros valores de x: ');
    fprintf('%.2f ', x(1:min(5,length(x))));
    fprintf('\n');
    fprintf('Primeros valores de y: ');
    fprintf('%.2f ', y(1:min(5,length(y))));
    fprintf('\n');
else
    fprintf('ERROR: No hay datos válidos después de la conversión!\n');
    fprintf('Valores originales de x (primeros 5): ');
    fprintf('%s ', string(df{mask_full, x_name}(1:min(5,height(df)))));
    fprintf('\n');
    fprintf('Valores originales de y (primeros 5): ');
    fprintf('%s ', string(df{mask_full, y_name}(1:min(5,height(df)))));
    fprintf('\n');
    return;
end

model_full = fit_ols_xy(x, y);

fprintf('\nParámetros (todos los datos):\n');
fprintf('Intercepto: %.4f\n', model_full.beta(1));
fprintf('Pendiente: %.4f\n', model_full.beta(2));

MSE_full = sum(model_full.resid.^2) / model_full.df_resid;
sigma_full = sqrt(MSE_full);
fprintf('Sigma (EE de la regresión) = %.6f\n', sigma_full);


figure;
scatter(x, y, 'DisplayName', 'Datos');
hold on;
xx = linspace(min(x), max(x), 200);
y_hat = model_full.beta(1) + model_full.beta(2) * xx;
plot(xx, y_hat, 'DisplayName', 'Recta ajustada');
xlabel('Valor catastral (miles de pesos)');
ylabel('Renta mensual (múltiplos de $5)');
title('RLS (todos): Renta ~ Valor');
legend('Location', 'best');
grid on;
grid minor;

plot_path = fullfile(PLOTS_DIR, 'full_scatter_line.png');
saveas(gcf, plot_path, 'png');
fprintf('Figura guardada en: %s\n', plot_path);


figure;
scatter(x, model_full.resid);
hold on;
yline(0, '--', 'LineWidth', 1);
xlabel('Valor catastral (miles de pesos)');
ylabel('Residuales');
title('Residuales vs x (todos)');
grid on;
grid minor;

resid_path = fullfile(PLOTS_DIR, 'full_resid_vs_x.png');
saveas(gcf, resid_path, 'png');
fprintf('Figura guardada en: %s\n', resid_path);

%%
fprintf('\n(b) ANOVA y significancia — todos los datos\n');
X_full = [ones(n_full, 1), x];
model_full = fit_ols_xy(x, y);


SS_total = sum((y - mean(y)).^2);
SS_model = model_full.ess;  
SS_resid = model_full.ssr;  
df_model = model_full.df_model;
df_resid = model_full.df_resid;
df_total = df_model + df_resid;

MS_model = SS_model / df_model;
MS_resid = SS_resid / df_resid;
F_stat = MS_model / MS_resid;
p_value = 1 - fcdf(F_stat, df_model, df_resid);


anova_data = array2table([df_model, df_resid, df_total; ...
                          SS_model, SS_resid, SS_total; ...
                          MS_model, MS_resid, NaN; ...
                          F_stat, NaN, NaN; ...
                          p_value, NaN, NaN]', ...
                         'VariableNames', {'df', 'sum_sq', 'mean_sq', 'F', 'PR_F'}, ...
                         'RowNames', {'x_valor_miles', 'Residual', 'Total'});
fprintf('\nANOVA (todos):\n');
disp(round(anova_data, 6));

F_full = F_stat;
p_full = p_value;
R2_full = model_full.rsquared;
fprintf('F = %.6f, p-valor = %.6f, R^2 = %.6f\n', F_full, p_full, R2_full);
fprintf('\nResumen del modelo (todos):\n');
fprintf('R^2: %.6f\n', model_full.rsquared);
fprintf('F-statistic: %.6f\n', F_stat);
fprintf('p-value: %.6f\n', p_value);

%%
fprintf('\n(c) Ajuste y significancia excluyendo y=0 ...\n');
mask_nz = df{:, y_name} ~= 0;

% Convert data for non-zero y values
if iscell(df{mask_nz, x_name})
    x_nz = str2double(df{mask_nz, x_name});
else
    x_nz = double(df{mask_nz, x_name});
end

if iscell(df{mask_nz, y_name})
    y_nz = str2double(df{mask_nz, y_name});
else
    y_nz = double(df{mask_nz, y_name});
end

% Check for NaN values and remove them
nan_mask_nz = ~isnan(x_nz) & ~isnan(y_nz);
x_nz = x_nz(nan_mask_nz);
y_nz = y_nz(nan_mask_nz);

n_nz = sum(nan_mask_nz);
fprintf('Datos válidos (sin y=0) después de remover NaN: %d\n', n_nz);

if n_nz == 0
    fprintf('ERROR: No hay datos válidos para y≠0 después de la conversión!\n');
    return;
end

model_nz = fit_ols_xy(x_nz, y_nz);

fprintf('Parámetros (sin y=0):\n');
fprintf('Intercepto: %.4f\n', model_nz.beta(1));
fprintf('Pendiente: %.4f\n', model_nz.beta(2));

MSE_nz = sum(model_nz.resid.^2) / model_nz.df_resid;
sigma_nz = sqrt(MSE_nz);
fprintf('Sigma (EE de la regresión, sin y=0) = %.6f\n', sigma_nz);


figure;
scatter(x_nz, y_nz, 'DisplayName', 'Datos (y>0)');
hold on;
xx = linspace(min(x_nz), max(x_nz), 200);
y_hat = model_nz.beta(1) + model_nz.beta(2) * xx;
plot(xx, y_hat, 'DisplayName', 'Recta ajustada (y>0)');
xlabel('Valor catastral (miles de pesos)');
ylabel('Renta mensual (múltiplos de $5)');
title('RLS (sin y=0): Renta ~ Valor');
legend('Location', 'best');
grid on;
grid minor;

plot_path2 = fullfile(PLOTS_DIR, 'nz_scatter_line.png');
saveas(gcf, plot_path2, 'png');
fprintf('Figura guardada en: %s\n', plot_path2);


figure;
scatter(x_nz, model_nz.resid);
hold on;
yline(0, '--', 'LineWidth', 1);
xlabel('Valor catastral (miles de pesos)');
ylabel('Residuales');
title('Residuales vs x (sin y=0)');
grid on;
grid minor;

resid_path2 = fullfile(PLOTS_DIR, 'nz_resid_vs_x.png');
saveas(gcf, resid_path2, 'png');
fprintf('Figura guardada en: %s\n', resid_path2);

X_nz = [ones(length(x_nz), 1), x_nz];
model_nz = fit_ols_xy(x_nz, y_nz);


SS_total_nz = sum((y_nz - mean(y_nz)).^2);
SS_model_nz = model_nz.ess;  
SS_resid_nz = model_nz.ssr;  
df_model_nz = model_nz.df_model;
df_resid_nz = model_nz.df_resid;
df_total_nz = df_model_nz + df_resid_nz;

MS_model_nz = SS_model_nz / df_model_nz;
MS_resid_nz = SS_resid_nz / df_resid_nz;
F_stat_nz = MS_model_nz / MS_resid_nz;
p_value_nz = 1 - fcdf(F_stat_nz, df_model_nz, df_resid_nz);


anova_data_nz = array2table([df_model_nz, df_resid_nz, df_total_nz; ...
                             SS_model_nz, SS_resid_nz, SS_total_nz; ...
                             MS_model_nz, MS_resid_nz, NaN; ...
                             F_stat_nz, NaN, NaN; ...
                             p_value_nz, NaN, NaN]', ...
                            'VariableNames', {'df', 'sum_sq', 'mean_sq', 'F', 'PR_F'}, ...
                            'RowNames', {'x_valor_miles', 'Residual', 'Total'});
fprintf('\nANOVA (sin y=0):\n');
disp(round(anova_data_nz, 6));

F_nz = F_stat_nz;
p_nz = p_value_nz;
R2_nz = model_nz.rsquared;
fprintf('F = %.6f, p-valor = %.6f, R^2 = %.6f\n', F_nz, p_nz, R2_nz);
fprintf('\nResumen del modelo (sin y=0):\n');
fprintf('R^2: %.6f\n', model_nz.rsquared);
fprintf('F-statistic: %.6f\n', F_stat_nz);
fprintf('p-value: %.6f\n', p_value_nz);

%%
fprintf('\n(d) Comparación de R^2 y guía de interpretación...\n');
fprintf('R^2 (todos)   = %.6f\n', R2_full);
fprintf('R^2 (sin y=0) = %.6f\n', R2_nz);
if R2_nz > R2_full
    fprintf('El ajuste mejora al remover y=0 (mayor R^2).\n');
elseif R2_nz < R2_full
    fprintf('El ajuste empeora al remover y=0 (menor R^2).\n');
else
    fprintf('R^2 es igual en ambos casos.\n');
end

%%
