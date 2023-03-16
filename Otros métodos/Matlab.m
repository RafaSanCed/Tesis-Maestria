function dydt = myODE(t, y, params)
    a1 = params(0);
    a2 = params(1);
    b1 = params(2);
    b2 = params(3);
    d1 = params(4);
    d2 = params(5);
    n = params(6);

    dydt = zeros(2,1);
    dydt(1) = a1 /(1+y(2)^n)-b1 * y(1) + d1;
    dydt(2) = a2 / (1+y(1)^n) - b2 * y(2) + d2;
end


function solution = solveODE(tspan, y0, params)
    [t, y] = ode45(@(t, y) myODE(t, y, params), tspan, y0);
    solution = [t, y];
end

function error = objective(params, tspan, y0, data)
    solution = solveODE(tspan, y0, params);
    model_output = solution(:, 2:end);
    error = sum(sum((model_output - data).^2));
end

% Datos experimentales (tiempo, y1, y2)
data = [0, 1, 0; 1, 0.5, 0.5; 2, 0.25, 0.75];

% Condiciones iniciales
y0 = [1; 0];

% Intervalo de tiempo
tspan = data(:, 1);

% Valores iniciales de los parámetros
params_init = [1, 1];

% Límites inferiores y superiores para los parámetros
lb = [0, 0];
ub = [10, 10];

% Estimar parámetros utilizando lsqcurvefit
[params_est, resnorm] = lsqcurvefit(@(params, tspan) objective(params, tspan, y0, data), params_init, tspan, data, lb, ub);

% Resultados
disp('Parámetros estimados:')
disp(params_est)