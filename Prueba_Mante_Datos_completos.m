% ================================================================
% SCRIPT: RVM con SparseBayes - Sensores numéricos filtrados
% ================================================================

% 0. Cargar librería RVM
addpath('C:\Users\eduar\Downloads\SB2_Release_200\SB2_Release_200');
savepath;
which SparseBayes;

% 1. Cargar datos de entrenamiento (hoja "train")
opts = detectImportOptions('train_analysis.xlsx', 'Sheet', 'train');
opts.VariableNamingRule = 'modify';
train = readtable('train_analysis.xlsx', opts);

% 2. Cargar archivo de prueba completo
test = readtable('test_completo.xlsx');
test.Properties.VariableNames = matlab.lang.makeValidName(test.Properties.VariableNames);

% 3. Identificar columnas de sensores (Sensor1, Sensor2, ...)
sensor_cols_train = train.Properties.VariableNames(contains(train.Properties.VariableNames, 'Sensor'));
sensor_cols_test = test.Properties.VariableNames(contains(test.Properties.VariableNames, 'Sensor'));
columnas_comunes = intersect(sensor_cols_train, sensor_cols_test);
columnas_entrada = setdiff(columnas_comunes, {'Sensor17'});  % Excluir la salida

% 4. Filtrar solo columnas numéricas válidas
tipos_validos = varfun(@class, test(:, columnas_entrada), 'OutputFormat', 'cell');
columnas_numericas = columnas_entrada(strcmp(tipos_validos, 'double'));

% 5. Construir matrices de entrada y salida
X_train = table2array(train(:, columnas_numericas));
y_train = table2array(train(:, 'Sensor17'));

X_test = table2array(test(:, columnas_numericas));
y_test = table2array(test(:, 'Sensor17'));

% 6. Entrenamiento con tiempo
tic;
model = SparseBayes('Gaussian', X_train, y_train);
tiempo_entrenamiento = toc;
fprintf('⏱️ Tiempo de entrenamiento: %.4f segundos\n', tiempo_entrenamiento);

% 7. Predicción con tiempo
relevant_vectors = model.Relevant;
weights = model.Value;
X_relevant = X_test(:, relevant_vectors);

tic;
y_pred = X_relevant * weights;
tiempo_prediccion = toc;
fprintf('⏱️ Tiempo de predicción: %.4f segundos\n', tiempo_prediccion);

% 8. Evaluación
rmse = sqrt(mean((y_test - y_pred).^2));
fprintf('📉 RMSE del modelo: %.4f\n', rmse);

fprintf('📌 Sensores relevantes utilizados:\n');
disp(columnas_numericas(relevant_vectors));

% 9. Gráfica: Todos los datos
figure;
plot(y_test, 'b', 'LineWidth', 1); hold on;
plot(y_pred, 'r--', 'LineWidth', 1);
legend('Valor Real', 'Predicción RVM');
xlabel('Índice de Muestra');
ylabel('Sensor17');
title('Comparación: Real vs Predicción (todos los datos)');
grid on;

% 10. Gráfica: Primeras 500 muestras
figure;
n = min(500, length(y_test));
plot(y_test(1:n), 'b', 'LineWidth', 1.5); hold on;
plot(y_pred(1:n), 'r--', 'LineWidth', 1.5);
legend('Valor Real', 'Predicción RVM');
xlabel('Índice de Muestra');
ylabel('Sensor17');
title('Predicción RVM (primeras 500 muestras)');
grid on;
