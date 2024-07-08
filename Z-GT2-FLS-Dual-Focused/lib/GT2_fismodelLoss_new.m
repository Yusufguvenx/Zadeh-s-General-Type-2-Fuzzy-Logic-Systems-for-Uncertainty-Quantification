function [loss, gradients, yPred_lower, yPred_upper, yPred] = GT2_fismodelLoss_new(x, number_inputs, y,number_outputs, number_mf, mbs, learnable_parameters, output_type, input_mf_type, input_type,type_reduction_method,u,alpha, delta, alpha_rev)

[yPred_lower1, yPred_upper1, ~] = GT2_fismodel_LA1_new(x, number_mf, number_inputs, number_outputs, mbs, learnable_parameters, output_type, input_mf_type, input_type, type_reduction_method, u, alpha, delta, alpha_rev);


yPred_lower = yPred_lower1(1, :, :);
yPred_upper = yPred_upper1(1, :, :);

yPred = (yPred_lower + yPred_upper) / 2;

loss_tilted = tilted_loss(y, yPred_lower1, yPred_upper1, 0, 1, mbs, alpha);


loss = loss_tilted;

gradients = dlgradient(loss, learnable_parameters);


end