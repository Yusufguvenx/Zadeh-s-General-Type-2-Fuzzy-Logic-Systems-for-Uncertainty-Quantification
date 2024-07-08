function [output_lower, output_upper, output_mean] = GT2_fismodel_LA1_new(x, number_mf, number_inputs, number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha, delta, alpha_rev)


fuzzified = T2_matrix_fuzzification_layer(x, input_mf_type,input_type, learnable_parameters, number_mf, number_inputs, mbs);



[smf_fuzzified_lower,smf_fuzzified_upper] = GT2_matrix_fuzzification_layer(fuzzified, alpha, learnable_parameters, delta);
[firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(smf_fuzzified_lower, smf_fuzzified_upper, "product");
firestrength_lower = permute(firestrength_lower, [1, 4, 3, 2]);
firestrength_upper = permute(firestrength_upper, [1, 4, 3, 2]);
[output_lower, output_upper, output_mean] = GT2_defuzzification_layer(x, firestrength_lower, firestrength_upper, learnable_parameters,number_outputs, output_type,type_reduction_method,mbs,number_mf,u);

end

