function [x_norm,minimum,range] = max_min_norm(x)
minimum = min(x);
range = max(x) - min(x);
x_norm = (x-minimum)./range;
end