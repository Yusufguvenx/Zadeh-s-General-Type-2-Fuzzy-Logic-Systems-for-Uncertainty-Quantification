function [loss, gradients, yPred_lower, yPred_upper, yPred] = GT2_fismodelLoss(x, number_inputs, y,number_outputs, number_mf, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u, alpha, delta, alpha_rev)

[yPred_lower, yPred_upper, yPred] = GT2_fismodel_LA1_new(x, number_mf, number_inputs, number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha, delta, alpha_rev);


loss = log_cosh_loss(yPred, y, mbs);


loss_tilted1 = tilted_loss(y, yPred_lower, yPred_upper, 0.005, 0.995, mbs);


loss = loss + loss_tilted1;

gradients = dlgradient(loss, learnable_parameters);


end