function x_denorm = zscore_denorm(x_norm,mean,std)
x_denorm = (x_norm.*std)+mean;
end
