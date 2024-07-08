function x = Quantile_Calc(y, y_q)
i_l = y<y_q;

n = length(y);

n_l = sum(i_l);

x = n_l / n;

end
