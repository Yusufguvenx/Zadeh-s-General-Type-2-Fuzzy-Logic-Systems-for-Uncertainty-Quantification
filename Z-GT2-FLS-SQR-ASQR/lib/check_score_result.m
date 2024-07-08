function check_scr = check_score_result(Test, y_true, number_mf, number_inputs, number_outputs, best_learnable_parameters, output_membership_type, input_membership_type, input_type, type_reduction_method, u, alpha_rev)

q_list = 0.01:0.01:0.99;
delta = "train";


check_list = [];
for i = 1:length(q_list)
    [~, ~, yPred_Test_q1] = GT2_fismodel_LA1_new(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), best_learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, q_list(i), delta, alpha_rev);
    q_level = permute(yPred_Test_q1, [1 3 2]);
    score_per_q = tilted_loss(y_true, q_level, 0, 0, length(Test.inputs), q_list(i));
    
    check_list = [check_list score_per_q];

end

check_scr = mean(check_list);



end

