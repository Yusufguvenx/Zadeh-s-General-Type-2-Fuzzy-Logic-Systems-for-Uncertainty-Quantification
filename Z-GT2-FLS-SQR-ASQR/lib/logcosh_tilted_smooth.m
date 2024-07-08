function loss = logcosh_tilted_smooth(y, ypred, alpha, mbs)

    alpha_as_quantile = permute(alpha, [1 2 4 3]); % 1 by 1 by mbs
    diff = y - ypred;

    loss = 0.05*log(cosh(10*diff)) + (alpha_as_quantile - 0.5).*diff;
    loss = 1/mbs*sum(loss);


end