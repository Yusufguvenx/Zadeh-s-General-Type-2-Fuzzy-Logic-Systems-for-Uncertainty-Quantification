function x = PINAW(y, y_l, y_u)
n = length(y_l);
r = max(y) - min(y);
x = sum(y_u-y_l)*1/(n*r);
end
