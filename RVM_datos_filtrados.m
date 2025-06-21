% ================================================
% SCRIPT: RVM con SparseBayes - Medición de tiempo
% ================================================

% 0. Cargar librería RVM
addpath('C:\Users\eduar\Downloads\SB2_Release_200\SB2_Release_200');
savepath;
which SparseBayes;

% 1. Cargar datos desde Excel
train = readtable('train.xlsx');
test = readtable('test.xlsx');

% 2. Seleccionar características (X) y variable objetivo (y)
X_train = table2array(train(:, {'Sensor3', 'Sensor4', 'Sensor9', ...
                                'Sensor11', 'Sensor14', 'Sensor15'}));
y_train = table2array(train(:, 'Sensor17'));

X_test = table2array(test(:, {'Sensor3', 'Sensor4', 'Sensor9', ...
                              'Sensor11', 'Sensor14', 'Sensor15'}));
y_test = table2array(test(:, 'Sensor17'));  % Solo para evaluación

% 3. Entrenar el modelo RVM con cronómetro
tic;  % Iniciar medición de tiempo
model = SparseBayes('Gaussian', X_train, y_train);
tiempo_entrenamiento = toc;  % Finalizar medición

fprintf('⏱️ Tiempo de entrenamiento (datos reducidos): %.4f segundos\n', tiempo_entrenamiento);

% 4. Predicción usando vectores relevantes
relevant_vectors = model.Relevant;
weights = model.Value;
X_relevant = X_test(:, relevant_vectors);

y_pred = X_relevant * weights;

% 5. Evaluación del modelo (RMSE)
rmse = sqrt(mean((y_test - y_pred).^2));
fprintf('📉 RMSE del modelo RVM: %.4f\n', rmse);

% 6. Graficar comparación completa
figure;
plot(y_test, 'b', 'LineWidth', 1); hold on;
plot(y_pred, 'r--', 'LineWidth', 1);
legend('Valor Real', 'Predicción RVM');
xlabel('Índice de Muestra');
ylabel('Sensor17');
title('Comparación: Valor real vs Predicción RVM (todos los datos)');
grid on;

% 7. Gráfica: Primeras 500 muestras
figure;
n = min(500, length(y_test));
plot(y_test(1:n), 'b', 'LineWidth', 1.5); hold on;
plot(y_pred(1:n), 'r--', 'LineWidth', 1.5);
legend('Valor Real', 'Predicción RVM');
xlabel('Índice de Muestra');
ylabel('Sensor17');
title('Predicción RVM (primeras 500 muestras)');
grid on;
