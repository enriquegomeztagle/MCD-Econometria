try
    df = readtable('../data/cableTV.xlsx', 'Format', 'auto');
    fprintf('File read successfully\n');
    read_success = true;
catch ME
    fprintf('Error while reading the file: %s\n', ME.message);
    read_success = false;
end

if read_success
    fprintf('--------------------------------\n');
    fprintf('Dataframe head:\n');
    disp(df(1:min(10, height(df)), :));
end

if read_success
    fprintf('--------------------------------\n');
    fprintf('Dataframe info:\n');
    fprintf('Shape: %d x %d\n', height(df), width(df));
    fprintf('Variable names: ');
    disp(df.Properties.VariableNames);
end

if read_success
    fprintf('--------------------------------\n');
    fprintf('Dataframe describe:\n');
    numeric_vars = varfun(@isnumeric, df, 'output', 'uniform');
    numeric_cols = df.Properties.VariableNames(numeric_vars);
    if ~isempty(numeric_cols)
        summary(df(:, numeric_cols));
    end
end

if read_success
    fprintf('--------------------------------\n');
    fprintf('Dataframe shape:\n');
    fprintf('%d x %d\n', height(df), width(df));
end

if read_success
    fprintf('--------------------------------\n');
    fprintf('Dataframe missing values:\n');
    for i = 1:width(df)
        var_name = df.Properties.VariableNames{i};
        if isnumeric(df{:, i})
            missing_count = sum(isnan(df{:, i}));
        else
            missing_count = sum(ismissing(df{:, i}));
        end
        fprintf('%s: %d\n', var_name, missing_count);
    end
    
    fprintf('--------------------------------\n');
    fprintf('Dataframe unique values:\n');
    for i = 1:width(df)
        var_name = df.Properties.VariableNames{i};
        if isnumeric(df{:, i})
            unique_count = length(unique(df{:, i}(~isnan(df{:, i}))));
        else
            unique_count = length(unique(df{:, i}(~ismissing(df{:, i}))));
        end
        fprintf('%s: %d\n', var_name, unique_count);
    end
end

if read_success
    quant_vars = {'adultos', 'ninos', 'teles', 'tvtot', 'renta', 'valor'};
    
    if ~exist('../plots/matlab', 'dir')
        mkdir('../plots/matlab');
    end
    
    for i = 1:length(quant_vars)
        var = quant_vars{i};
        if ismember(var, df.Properties.VariableNames)
            values = df.(var);
            label = var;
            
            if strcmp(var, 'valor')
                values = values / 1000;
                label = [label ' (thousands of pesos)'];
            end
            
            figure('Visible', 'off');
            histogram(values, 'EdgeColor', 'black');
            xlabel(label);
            ylabel('Frequency');
            title(['Frequency of ' label]);
            grid on;
            grid minor;
            
            saveas(gcf, ['../plots/matlab/frequency' var '.png']);
            close(gcf);
        end
    end
end

if read_success
    cat_vars = {'colonia', 'tipo'};
    
    for i = 1:length(cat_vars)
        var = cat_vars{i};
        if ismember(var, df.Properties.VariableNames)
            fprintf('=== Frequency of ''%s'' ===\n', var);
            [unique_vals, ~, idx] = unique(df.(var));
            counts = accumarray(idx, 1);
            
            [counts_sorted, sort_idx] = sort(counts, 'descend');
            unique_vals_sorted = unique_vals(sort_idx);
            
            for j = 1:length(unique_vals_sorted)
                fprintf('%s: %d\n', string(unique_vals_sorted(j)), counts_sorted(j));
            end
            fprintf('\n');
        end
    end
end
