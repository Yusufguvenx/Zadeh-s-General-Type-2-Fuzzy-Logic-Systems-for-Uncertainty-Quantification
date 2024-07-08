function [loss, gradients, yPred_lower, yPred_upper, yPred] = GT2_fismodelLoss(x, number_inputs, y,number_outputs, number_mf, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u, alpha, delta, alpha_rev)

[yPred_lower, yPred_upper, yPred] = GT2_fismodel_LA1_new(x, number_mf, number_inputs, number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha, delta, alpha_rev);



loss_tilted1 = tilted_loss(y, yPred, 0, 1, mbs, alpha);


loss = loss_tilted1;

gradients = dlgradient(loss, learnable_parameters);


end