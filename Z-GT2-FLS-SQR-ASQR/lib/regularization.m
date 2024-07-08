function reg_term = regularization(y, ypred, alpha)

    y = permute(y, [1 3 2]);
    ypred = permute(ypred, [1 3 2]);

    Quantile_level = Quantile_Calc(y, ypred);

    err = alpha - Quantile_level;
    
    rSquare = 1 - sum(err.^2) / sum(alpha - mean(alpha));

    reg_term = rSquare;

end