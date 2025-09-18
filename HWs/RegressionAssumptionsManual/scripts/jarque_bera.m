rng(42);
n = 300;
variable_independiente = randn(n,1);
variable_dependiente  = 5 + 2*randn(n,1) + randn(n,1);

tbl = table(variable_dependiente, variable_independiente);


mdl = fitlm(tbl, 'variable_dependiente ~ variable_independiente');
residuos = mdl.Residuals.Raw;

alpha = 0.05;
try
   [h, p, jbstat] = jbtest(residuos, alpha, 1e-3);
 catch ME
    warning("jbtest con Monte Carlo falló (%s). Uso p-valor asintótico chi2(2).", ME.message);
    [~, ~, jbstat] = jbtest(residuos, alpha);
    p = 1 - chi2cdf(jbstat, 2);
 end

fprintf('JB: %.4f\np-valor: %.6f\n', jbstat, p);
if h == 1
    disp('Rechazamos H0: los residuos no son normales.');
else
    disp('No rechazamos H0: no hay evidencia contra la normalidad.');
end
