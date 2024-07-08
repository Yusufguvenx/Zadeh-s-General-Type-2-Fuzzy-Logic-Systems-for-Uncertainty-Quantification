function loss = logcosh_tilted(y, ypred, alpha, mbs)

    alpha_as_quantile = permute(alpha, [1 2 4 3]); % 1 by 1 by mbs

    diff = 1*(y - ypred);
    loss = (alpha_as_quantile - 0.5).*diff + (0.5 / 0.7)*log(cosh(0.7*diff)) + 0.4;
    loss = sum(loss);

end