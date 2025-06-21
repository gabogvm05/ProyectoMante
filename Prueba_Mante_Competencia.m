% ================================================================
% SCRIPT: RVM con SparseBayes - Sensores numéricos filtrados (Competencia)
% ================================================================

% 0. Cargar librería RVM
addpath('C:\Users\eduar\Downloads\SB2_Release_200\SB2_Release_200');
savepath;
which SparseBayes;

% 1. Cargar datos de entrenamiento (hoja "train")
opts = detectImportOptions('train_analysis.xlsx', 'Sheet', 'train');
opts.VariableNamingRule = 'modify';
train = readtable('train_analysis.xlsx', opts);

% 2. Cargar archivo de competencia (sin encabezados)
test_raw = readmatrix('Datos para validación de modelo y competencia.xlsx');

% 3. Crear nombres de columnas automáticamente
num_columnas = size(test_raw, 2);

% 5 primeras columnas fijas
columnas = ["UnitNumber", "TimeInCycles", "OpSetting1", "OpSetting2", "OpSetting3"];

% Agregar dinámicamente los nombres de sensores
for i = 1:(num_columnas - 5)
    columnas(end+1) = "Sensor" + i;
end

% Convertir a tabla con nombres generados
test = array2table(test_raw, 'VariableNames', columnas);

% 4. Identificar columnas de sensores
sensor_cols_train = train.Properties.VariableNames(contains(train.Properties.VariableNames, 'Sensor'));
sensor_cols_test = test.Properties.VariableNames(contains(test.Properties.VariableNames, 'Sensor'));
columnas_comunes = intersect(sensor_cols_train, sensor_cols_test);
columnas_entrada = setdiff(columnas_comunes, {'Sensor17'});  % Excluir la salida

% 5. Filtrar solo columnas numéricas válidas
tipos_validos = varfun(@class, test(:, columnas_entrada), 'OutputFormat', 'cell');
columnas_numericas = columnas_entrada(strcmp(tipos_validos, 'double'));

% 6. Construir matrices de entrada y salida
X_train = table2array(train(:, columnas_numericas));
y_train = table2array(train(:, 'Sensor17'));
X_test = table2array(test(:, columnas_numericas));

% Si no hay y_test en este archivo (es decir, no hay Sensor17), evita evaluación
tiene_y_test = ismember('Sensor17', test.Properties.VariableNames);
if tiene_y_test
    y_test = table2array(test(:, 'Sensor17'));
end

% 7. Entrenamiento con tiempo
tic;
model = SparseBayes('Gaussian', X_train, y_train);
tiempo_entrenamiento = toc;
fprintf('⏱️ Tiempo de entrenamiento: %.4f segundos\n', tiempo_entrenamiento);

% 8. Predicción con tiempo
relevant_vectors = model.Relevant;
weights = model.Value;
X_relevant = X_test(:, relevant_vectors);

tic;
y_pred = X_relevant * weights;
tiempo_prediccion = toc;
fprintf('⏱️ Tiempo de predicción: %.4f segundos\n', tiempo_prediccion);

% 9. Evaluación (si hay y_test)
if tiene_y_test
    rmse = sqrt(mean((y_test - y_pred).^2));
    fprintf('📉 RMSE del modelo: %.4f\n', rmse);
end

fprintf('📌 Sensores relevantes utilizados:\n');
disp(columnas_numericas(relevant_vectors));

% 10. Gráfica: Todos los datos (solo si existe y_test)
if tiene_y_test
    figure;
    plot(y_test, 'b', 'LineWidth', 1); hold on;
    plot(y_pred, 'r--', 'LineWidth', 1);
    legend('Valor Real', 'Predicción RVM');
    xlabel('Índice de Muestra');
    ylabel('Sensor17');
    title('Comparación: Real vs Predicción (todos los datos)');
    grid on;

    % 11. Gráfica: Primeras 500 muestras
    figure;
    n = min(500, length(y_test));
    plot(y_test(1:n), 'b', 'LineWidth', 1.5); hold on;
    plot(y_pred(1:n), 'r--', 'LineWidth', 1.5);
    legend('Valor Real', 'Predicción RVM');
    xlabel('Índice de Muestra');
    ylabel('Sensor17');
    title('Predicción RVM (primeras 500 muestras)');
    grid on;
end
