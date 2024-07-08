function [smf_fuzzified_lower,smf_fuzzified_upper] = GT2_matrix_fuzzification_layer(fuzzified, alpha, learnable_parameters, delta)

lower_sigma = (fuzzified/sqrt(-2*log(alpha(1)))).*sigmoid(learnable_parameters.smf.lower_sigma);
upper_sigma = (1-fuzzified)/sqrt(-2*log(alpha(1))).*sigmoid(learnable_parameters.smf.upper_sigma);

smf_fuzzified_lower = (fuzzified - sqrt(-2*log(alpha)).*abs(lower_sigma));

smf_fuzzified_upper = (fuzzified + sqrt(-2*log(alpha)).*abs(upper_sigma));


end

