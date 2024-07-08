function tilted_loss = tilted_loss(y, ypred, q1, q2, mbs, alpha)

y_to_be_used = ypred;


alpha_as_quantile = permute(alpha, [1 2 4 3]); % 1 by 1 by mbs

tilted_loss = 1/mbs*sum(max(alpha_as_quantile.*(y-y_to_be_used), (alpha_as_quantile-1).*(y-y_to_be_used)), "all");



end

