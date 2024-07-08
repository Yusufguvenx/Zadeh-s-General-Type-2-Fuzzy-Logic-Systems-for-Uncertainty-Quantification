function tilted_loss = tilted_loss(y, y_lower, y_upper, q1, q2, mbs, alpha)


lower_loss = 1/mbs*(sum(max(q1*(y-y_lower), (q1-1)*(y-y_lower))));
upper_loss = 1/mbs*(sum(max(q2*(y-y_upper), (q2-1)*(y-y_upper))));

% tilted_loss = 1/mbs*sum(max(alpha_as_quantile.*(y-y_to_be_used), (alpha_as_quantile-1).*(y-y_to_be_used)), "all");

% alpha_as_quantile=0.;
% 
% tilted_loss = tilted_loss+sum(max(alpha_as_quantile.*(y-y_to_be_used), (alpha_as_quantile-1).*(y-y_to_be_used)), "all");
% 
% alpha_as_quantile=0.;
% 
% tilted_loss = tilted_loss+sum(max(alpha_as_quantile.*(y-y_to_be_used), (alpha_as_quantile-1).*(y-y_to_be_used)), "all");


% tilted_loss = 1/mbs*(sum(sum(max(alpha_as_quantile.*(y-y_to_be_used), (alpha_as_quantile-1).*(y-y_to_be_used)))));
% upper_loss = 1/mbs*(sum(max(q2*(y-y_to_be_used), (q2-1)*(y-y_to_be_used))));


tilted_loss = lower_loss + upper_loss;

end

