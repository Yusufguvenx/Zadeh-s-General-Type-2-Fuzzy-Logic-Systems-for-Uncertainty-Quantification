function alpha_values = alpha_generator(total_min_values, total_scaling_values, total_cell, rest, error, resultArray)

    cellArray = cellfun(@(dims) rand(1, dims), num2cell(total_cell), 'UniformOutput', false);
    
        % Define a function to multiply each matrix by the corresponding element of the vector
    multiplyFunction = @(matrix, min_val, scaling_val) min_val + matrix * scaling_val;
    %multi_func = @(error, resultArray) sum(error.*resultArray);
%     area = cellfun(multi_func, error, resultArray, 'UniformOutput', false); % area has found // burada kaldım.





%     min_index = total_min_values*100;
%     max_index = min_index + total_scaling_values*100;

    % Use cellfun to apply the function to each cell of the cell array

    if(isempty(error))
        alpha_values = [rand(1, rest)];
    else
        alpha_values = cellfun(multiplyFunction, cellArray, num2cell(total_min_values), num2cell(total_scaling_values), 'UniformOutput', false);
        alpha_values = horzcat(alpha_values{:});
        alpha_values = [rand(1, rest), alpha_values];
    end

end