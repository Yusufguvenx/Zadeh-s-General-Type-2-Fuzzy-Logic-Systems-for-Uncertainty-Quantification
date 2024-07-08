function [X0, targets]  = create_mini_batch(X, yTrue, minibatch_size)

shuffle_idx = randperm(size(X, 3), minibatch_size);

X0 = X(:, :, shuffle_idx);
targets = yTrue(:, :, shuffle_idx);

if canUseGPU
    X0 = gpuArray(X0);
    targets = gpuArray(targets);
end

end