function [val_loss, total_min_values, total_scaling_values, total_cell, rest, error, resultArray] = val_loss_2(y, ypred, ypred_opp, mbs, alpha, eps)

y_stacked = repmat(y, size(alpha, 4), 1);
idx_under = y_stacked <= ypred;
coverage = mean(idx_under, 3);

alpha_to_be_used = permute(alpha, [4 3 2 1]);

[minValues, minValues_zeros,scaling_factor, scaling_factor_zeros, cellLength, cellLength_zeros, error, resultArray] = createGroups(coverage, alpha_to_be_used, eps);


error = split_vector(extractdata(error));
error_sum = cellfun(@sum, error);

error_normalized = error_sum./sum(error_sum);


cellLength = floor(mbs*error_normalized);
total_cell = [cellLength];
rest = mbs - sum(total_cell);

total_min_values = [minValues];
total_scaling_values = [scaling_factor];


val_loss = mean(abs(coverage-alpha_to_be_used));
% val_loss = loss;
end

