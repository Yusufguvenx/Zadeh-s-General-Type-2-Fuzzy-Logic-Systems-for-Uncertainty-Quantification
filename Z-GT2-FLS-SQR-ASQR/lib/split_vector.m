
function resultingSubarrays = split_vector(arr)
    nonZeroIndices = find(arr ~= 0);
    
    if isempty(nonZeroIndices)
        resultingSubarrays = {};
        return;
    end
    
    endIndices = [nonZeroIndices(find(diff(nonZeroIndices) > 1)); nonZeroIndices(end)];
    startIndices = [nonZeroIndices(1); nonZeroIndices(find(diff(nonZeroIndices) > 1) + 1)];
    
    resultingSubarrays = cellfun(@(start, stop) arr(start:stop)', num2cell(startIndices), num2cell(endIndices), 'UniformOutput', false);
    
    % Transpose the resulting cell array
    resultingSubarrays = resultingSubarrays;
end