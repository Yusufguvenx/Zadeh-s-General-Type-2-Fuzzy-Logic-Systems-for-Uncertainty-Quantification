function [smf_fuzzified_lower,smf_fuzzified_upper] = GT2_matrix_fuzzification_layer(fuzzified, alpha, learnable_parameters, mode)




lower_sigma = fuzzified/sqrt(-2*log(0.01)).*sigmoid(learnable_parameters.smf.lower_sigma);
upper_sigma = (1-fuzzified)/sqrt(-2*log(0.01)).*sigmoid(learnable_parameters.smf.upper_sigma);


if(mode=="train")
    alpha = permute(alpha, [1 2 4 3]);
% elseif(mode=="val")
%     alpha = alpha;
end

smf_fuzzified_lower = fuzzified - sqrt(-2*log(alpha)).*lower_sigma;

smf_fuzzified_upper = fuzzified + sqrt(-2*log(alpha)).*upper_sigma;


end

