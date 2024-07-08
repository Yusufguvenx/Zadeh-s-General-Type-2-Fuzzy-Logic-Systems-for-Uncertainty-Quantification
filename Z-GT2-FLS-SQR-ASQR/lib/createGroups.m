function [minValues, minValues_zeros, scaling_factor, scaling_factor_zeros, cellLengths, cellLengths_zeros, error, resultArray] = createGroups(coverage, alpha_to_be_used, eps)

    error = abs(coverage-alpha_to_be_used);
    cond = error >= eps;
    numbers = alpha_to_be_used.*cond;
    zeros_numbers = find(extractdata(numbers) == 0);
    numbers = find(extractdata(numbers) ~= 0);

    % Calculate differences between consecutive elements
    diffs = diff(numbers);
    diffs_zeros = diff(zeros_numbers);
    
    % Find indices where the difference is greater than 1
    splitIndices = find(diffs > 1);
    splitIndices_zeros = find(diffs_zeros > 1);
    
    % Add the first and last indices to split the array
    splitIndices = [0, splitIndices', numel(numbers)'];
    splitIndices_zeros = [0, splitIndices_zeros', numel(zeros_numbers)'];
    
    % Use arrayfun to create cell array of groups
    result = arrayfun(@(i) numbers(splitIndices(i) + 1:splitIndices(i + 1)), 1:numel(splitIndices) - 1, 'UniformOutput', false);
    result_zeros = arrayfun(@(i) zeros_numbers(splitIndices_zeros(i) + 1:splitIndices_zeros(i + 1)), 1:numel(splitIndices_zeros) - 1, 'UniformOutput', false);

    % Check the length of each group
    cellLengths = cellfun(@length, result);
    cellLengths_zeros = cellfun(@length, result_zeros);
    
    result = result(cellLengths>=1);
    result_zeros = result_zeros(cellLengths_zeros>=1);
    max_values = cellfun(@max, result)*0.01;
    min_values = cellfun(@min, result)*0.01;
    max_values_zeros = cellfun(@max, result_zeros)*0.01;
    min_values_zeros = cellfun(@min, result_zeros)*0.01;

    % Determine the number of pairs
    numPairs = min(length(max_values), length(min_values));
    numPairs_zeros = min(length(min_values_zeros), length(max_values_zeros));
    
    % Create a cell array of pairs using array operations
    myTuple = arrayfun(@(i) [min_values(i), max_values(i)], 1:numPairs, 'UniformOutput', false);
    pairsArray = vertcat(myTuple{:});

    myTuple_zeros = arrayfun(@(i) [min_values_zeros(i), max_values_zeros(i)], 1:numPairs_zeros, 'UniformOutput', false);
    pairsArray_zeros = vertcat(myTuple_zeros{:});

    if(isempty(pairsArray))
        minValues = 0;
        maxValues = 0;
    else
        minValues = pairsArray(:, 1);
        maxValues = pairsArray(:, 2);
    end


    if(isempty(pairsArray_zeros))
        minValues_zeros = 0;
        maxValues_zeros = 0;
    else
        minValues_zeros = pairsArray_zeros(:, 1);
        maxValues_zeros = pairsArray_zeros(:, 2);
    end
    % Specify the step size
    stepSize = 0.01;
    
    % Generating an array between min and max values with the specified step size
    resultArray = cellfun(@(minVal, maxVal) minVal:stepSize:maxVal, num2cell(minValues), num2cell(maxValues), 'UniformOutput', false);
    resultArray_zeros = cellfun(@(minVal, maxVal) minVal:stepSize:maxVal, num2cell(minValues_zeros), num2cell(maxValues_zeros), 'UniformOutput', false);


    cellLengths = cellLengths*stepSize;
    cellLengths = cellLengths';
    cellLengths_zeros = cellLengths_zeros*stepSize;
    cellLengths_zeros = cellLengths_zeros';
    scaling_factor = maxValues-minValues;
    scaling_factor_zeros = maxValues_zeros - minValues_zeros;

    error = (cond.*error);

end