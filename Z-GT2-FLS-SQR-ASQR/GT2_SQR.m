
% Here we represent learning of all quantile levels simultaneously using
% Z-GT2-FLS. We select R(number of rules) = 10 and number of epoch = 10000.
% We use validation during training if validation loss does not decrease
% for 200 epochs we early stop and take learnable parameters with lowest
% validation loss.
%
% Qauntile level is produced randomly from unifrom distribution during
% training for each sample point, This model is called SQR.
%
%
%
% Powerplant dataset is used in this example. Z-score normalization is
% applied.
%
%@inproceedings{NEURIPS2019_73c03186,
 %author = {Tagasovska, Natasa and Lopez-Paz, David},
 %booktitle = {Advances in Neural Information Processing Systems},
 %editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
 %pages = {},
 %publisher = {Curran Associates, Inc.},
 %title = {Single-Model Uncertainties for Deep Learning},
 %url = {https://proceedings.neurips.cc/paper_files/paper/2019/file/73c03186765e199c116224b68adc5fa0-Paper.pdf},
 %volume = {32},
 %year = {2019}
%}


%%
clc;clear;
close all;
seed = 0;
rng(seed)
save_flag = "down"

%% PP

dataset_name = "powerplant";
load("/home/yusuf/Desktop/fuzzy/GT2_Github/dataset/Datasets/CCPP.mat")

data = [x y];

% training_num = 6697;
mbs = 64;
lr = 1e-2;


%%
current_path = pwd;
number_mf = 10; % number of rules == number of membership functions

number_inputs = min(size(x));
number_outputs = min(size(y));

number_of_epoch = 10000;

input_membership_type = "gaussmf";

input_type ="H";
% input_type ="S";
% input_type ="HS";

output_membership_type = "linear";

type_reduction_method = "KM";


if type_reduction_method == "KM"
u = int2bit(0:(2^number_mf)-1,number_mf);
% u = u(:, 2:end);
else
    u = 0;
end

delta = struct;
delta.delta1 = dlarray(zeros(1, 1));
delta.delta4 = dlarray(ones(1, 1));


type_reduction_method_list = ["KM"];
% type_reduction_method_list = ["NT"];


% alpha = rand(number_of_epoch, mbs); % alphacut representation uniform
% alpha = permute(alpha, [4 3 1 2]);

alpha_rev = 0;
% eps = 0.05;

gradDecay = 0.9;
sqGradDecay = 0.999;

PI_values = [];
test_results = [];
PINAW_values = [];
ece_values = [];
check_score = [];

averageGrad = [];
averageSqGrad = [];

plotFrequency = 10;



learnRate = lr;

averageGrad = [];
averageSqGrad = [];



close all
%%
rng(seed)
%%

if type_reduction_method == "KM"
    u = int2bit(0:(2^number_mf)-1,number_mf);
else
    u = 0;
end
%% Normalization upfront ------------------------------

  [trainInd, testInd] = dividerand(size(data, 1), 0.9, 0.1, 0);
  X_train_init = x(trainInd, :);
  y_train_init = y(trainInd);

  X_test = x(testInd, :);
  y_test = y(testInd);

  [trainInd_new, valInd] = dividerand(size(X_train_init, 1), 0.8, 0.2, 0);

  X_train = X_train_init(trainInd_new, :);
  y_train = y_train_init(trainInd_new);

  X_val = X_train_init(valInd, :);
  y_val = y_train_init(valInd);

  [X_train_scaled, mu_x, sigma_x] = zscore(X_train);
  [y_train_scaled, mu_y, sigma_y] = zscore(y_train);

  X_val_scaled = (X_val - mu_x) ./ sigma_x;
  y_val_scaled = (y_val - mu_y) ./ sigma_y;

  X_test_scaled = (X_test - mu_x) ./ sigma_x;
  y_test_scaled = (y_test - mu_y) ./ sigma_y;

  X_val_scaled(isnan(X_val_scaled)) = 0;
  X_test_scaled(isnan(X_test_scaled)) = 0;
%% ------------------------------




%training data
Train.inputs = reshape(X_train_scaled', [1, number_inputs, length(X_train_scaled)]); % traspose come from the working mechanism of the reshape, so it is a must
Train.outputs = reshape(y_train_scaled', [1, number_outputs, length(y_train_scaled)]);

Train.inputs = dlarray(Train.inputs);
Train.outputs = dlarray(Train.outputs);

%validation data

Val.inputs = reshape(X_val_scaled', [1, number_inputs, length(X_val_scaled)]); % traspose come from the working mechanism of the reshape, so it is a must
Val.outputs = reshape(y_val_scaled', [1, number_outputs, length(y_val_scaled)]);

Val.inputs = dlarray(Val.inputs);
Val.outputs = dlarray(Val.outputs);

%testing data
Test.inputs = reshape(X_test_scaled', [1, number_inputs, length(X_test_scaled)]);
Test.outputs = reshape(y_test_scaled', [1, number_outputs, length(y_test_scaled)]);


%% init

Learnable_parameters = initialize_Glorot_GT2(Train.inputs, input_type, Train.outputs,output_membership_type, number_mf);
prev_learnable_parameters = Learnable_parameters;


%% rng reset
rng(seed)

yTrue_train = reshape(Train.outputs, [1, max(size(Train.inputs))]);
yTrue_test = reshape(Test.outputs, [1, max(size(Test.inputs))]);
yTrue_val = reshape(Val.outputs, [1, max(size(Val.inputs))]);

%%

number_of_iter_per_epoch = floorDiv(length(X_train_scaled), mbs);

number_of_iter = number_of_epoch * number_of_iter_per_epoch;
global_iteration = 1;

patience = 200;
rbest = -inf;
wait = 0;

best_val_loss = inf;
%         alpha = rand(1, mbs);

for epoch = 1: number_of_epoch
    
    delta = "train";
    [batch_inputs, batch_targets] = create_mini_batch(Train.inputs, Train.outputs, length(X_train_scaled));

    for iter = 1:number_of_iter_per_epoch

        alpha = rand(1, mbs);
                        
        alpha_to_be_used = permute(alpha, [4 3 1 2]);

        [mini_batch_inputs, targets] = call_batch(batch_inputs, batch_targets,iter,mbs);


        [loss, gradients, yPred_train_lower, yPred_train_upper, yPred_train] = dlfeval(@GT2_fismodelLoss, mini_batch_inputs ,...
            number_inputs, targets,number_outputs, number_mf, mbs, Learnable_parameters, output_membership_type,...
            input_membership_type,input_type,type_reduction_method,u, alpha_to_be_used, delta, alpha_rev);
        
        [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(Learnable_parameters, gradients, averageGrad, averageSqGrad,...
            global_iteration, learnRate, gradDecay, sqGradDecay);



        global_iteration = global_iteration + 1;

    end


 %validation check   
if(mod(epoch, 1) == 0 || epoch==1)    

    alpha_val = 0.01:0.01:0.99;
    alpha_val = permute(alpha_val, [4, 3, 1, 2]);
    opp_alpha_val = 1 - alpha_val;

    delta = "val";
    
    [~, ~, yPred_val] = GT2_fismodel_LA1_new(Val.inputs, number_mf, number_inputs,number_outputs,length(Val.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha_val, delta, alpha_rev);
%             [~, ~, yPred_opp_val] = GT2_fismodel_LA1_new(Val.inputs, number_mf, number_inputs,number_outputs,length(Val.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, opp_alpha_val, delta, alpha_rev);
    [validation_loss, total_min_values, total_scaling_values, total_cell, rest, error, resultArray] = val_loss_2(Val.outputs, yPred_val, 0, mbs, alpha_val, eps);
   
    if validation_loss < best_val_loss
        best_val_loss = validation_loss;
        wait = 0;
        best_learnable_parameters = Learnable_parameters;
    else
        wait = wait + 1;
        if (wait >= patience)
            break
        end

    end

end
    alpha_q1 = ones(1, 1)*0.025;
    alpha_q2 = ones(1, 1)*0.975;

    %testing in each epoch for %95 coverage
    [yPred_test_lower1, yPred_test_upper1, yPred_test1] = GT2_fismodel_LA1_new(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), best_learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha_q1, delta, alpha_rev);
    [yPred_test_lower2, yPred_test_upper2, yPred_test2] = GT2_fismodel_LA1_new(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), best_learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha_q2, delta, alpha_rev);
    
    yPred_test1 = reshape(yPred_test1, [1, max(size(Test.inputs))]);

    yPred_test2 = reshape(yPred_test2, [1, max(size(Test.inputs))]);

    iter_plot_T2(epoch,plotFrequency,loss,yTrue_test, yPred_test1, yPred_test2, yPred_test_lower1);



    
    if(mod(epoch, 10) == 0)

        PI_test = PICP(yTrue_test, yPred_test1, yPred_test2);
    end


end
%% Inference

alpha_q1 = ones(1, 1)*0.025;
alpha_q2 = ones(1, 1)*0.975;


[~, ~, yPred_train_q1] = GT2_fismodel_LA1_new(Train.inputs, number_mf, number_inputs,number_outputs,length(Train.inputs), best_learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha_q1, delta, alpha_rev);
[~, ~, yPred_train_q2] = GT2_fismodel_LA1_new(Train.inputs, number_mf, number_inputs,number_outputs,length(Train.inputs), best_learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha_q2, delta, alpha_rev);
[ ~, ~, yPred_test_q1] = GT2_fismodel_LA1_new(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), best_learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha_q1, delta, alpha_rev);
 [~, ~, yPred_test_q2] = GT2_fismodel_LA1_new(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), best_learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha_q2, delta, alpha_rev);


yPred_test_q1 = reshape(yPred_test_q1, [1, max(size(Test.inputs))]);
yPred_test_q2 = reshape(yPred_test_q2, [1, max(size(Test.inputs))]);
yPred_train_q1 = reshape(yPred_train_q1, [1, max(size(Train.inputs))]);
yPred_train_q2 = reshape(yPred_train_q2, [1, max(size(Train.inputs))]);

Quantile_level_1 = Quantile_Calc(yTrue_test, yPred_test_q1);
Quantile_level_2 = Quantile_Calc(yTrue_test, yPred_test_q2);
Quantile_level_1_train = Quantile_Calc(yTrue_train, yPred_train_q1);
Quantile_level_2_train = Quantile_Calc(yTrue_train, yPred_train_q2);
% 
figure(5)
plot(yTrue_test, "rx"); hold on;
plot(yPred_test_q2, "b")
plot(yPred_test_q1, "Color", "#F80")
legend("yTrue", "\tau_u_p_p_e_r", "\tau_l_o_w_e_r")





PI_test = PICP(yTrue_test, yPred_test_q1, yPred_test_q2)

if (PI_test >= 92.5 && PI_test <= 100)
    PI_values = [PI_values, PI_test]
end



PI_train = PICP(yTrue_train, yPred_train_q1, yPred_train_q2);
PI_NAW_test = PINAW(yTrue_test, yPred_test_q1, yPred_test_q2);
PINAW_values = [PINAW_values, PI_NAW_test];

%% QQT Graph multiple quantiles for train

alphas = 0.01:0.01:0.99;

predicted_tau_levels = [];

for k = 1:length(alphas)

[~, ~, yPred_Train_q1] = GT2_fismodel_LA1_new(Train.inputs, number_mf, number_inputs,number_outputs,length(Train.inputs), best_learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alphas(k), delta, alpha_rev);
yPred_Train_q1 = reshape(yPred_Train_q1, [1, max(size(Train.inputs))]);
Quantile_level_1 = Quantile_Calc(yTrue_train, yPred_Train_q1);
predicted_tau_levels = [predicted_tau_levels Quantile_level_1];

end

figure(2);
plot(alphas,alphas, "rx"); hold on;
plot(alphas,predicted_tau_levels, "b")

% saveas(1, "training_qqt.fig")
r_train = 1 - sum((alphas - predicted_tau_levels).^2) / sum((alphas - mean(alphas)).^2);



%%

alpha_deneme = 0.01:0.01:0.99;
alpha_deneme = permute(alpha_deneme, [4, 3, 1, 2]);
delta = "val";

[~, ~, yPred_Test_q1] = GT2_fismodel_LA1_new(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), best_learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha_deneme, delta, alpha_rev);
yTrue_test1 = permute(yTrue_test, [1 3 2]);
y_stacked = repmat(yTrue_test1, 99, 1);
idx_under = y_stacked <= yPred_Test_q1;
coverage = mean(idx_under, 3);
average_ece_total = mean(abs(coverage'-alphas));

r_test_total = 1 - sum((alphas - predicted_tau_levels).^2) / sum((alphas - mean(alphas)).^2);


figure(7);
plot(alphas,alphas, "rx"); hold on;
plot(alphas,coverage', "b")

%% check_score_2
check_score_res = check_score_result(Test, yTrue_test, number_mf, number_inputs, number_outputs, best_learnable_parameters, output_membership_type, input_membership_type, input_type, type_reduction_method, u, alpha_rev); 
int_score = [];
%% interval score
int_score_res = interval_score_result(Test, yTrue_test, number_mf, number_inputs, number_outputs, best_learnable_parameters, output_membership_type, input_membership_type, input_type, type_reduction_method, u, alpha_rev); 
int_score = [int_score int_score_res];

%% QQT Graph multiple quantiles for test

alphas = 0.01:0.01:0.99;

predicted_tau_levels = [];
delta = "train";
for k = 1:length(alphas)

[~, ~, yPred_Test_q1] = GT2_fismodel_LA1_new(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), best_learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alphas(k), delta, alpha_rev);
yPred_Test_q1 = reshape(yPred_Test_q1, [1, max(size(Test.inputs))]);
Quantile_level_1 = Quantile_Calc(yTrue_test, yPred_Test_q1);
predicted_tau_levels = [predicted_tau_levels Quantile_level_1];

end

figure(3);
plot(alphas,alphas, "rx"); hold on;
plot(alphas,predicted_tau_levels, "b")

%     saveas(2, )

%     diff = sum(abs(predicted_tau_levels - alphas));

average_ece = mean(abs(predicted_tau_levels-alphas));
ece_values = [ece_values, average_ece];

r_test = 1 - sum((alphas - predicted_tau_levels).^2) / sum((alphas - mean(alphas)).^2);

%% QQT for val

alphas_val = 0.01:0.01:0.99;

predicted_tau_levels = [];

for k = 1:length(alphas_val)

[~, ~, yPred_Val_q1] = GT2_fismodel_LA1_new(Val.inputs, number_mf, number_inputs,number_outputs,length(Val.inputs), best_learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alphas_val(k), delta, alpha_rev);
yPred_Val_q1 = reshape(yPred_Val_q1, [1, max(size(Val.inputs))]);
Quantile_level_1 = Quantile_Calc(yTrue_val, yPred_Val_q1);
predicted_tau_levels = [predicted_tau_levels Quantile_level_1];

end


figure(6);
plot(alphas_val,alphas_val, "rx"); hold on;
plot(alphas_val,predicted_tau_levels, "b")


r_val = 1 - sum((alphas_val - predicted_tau_levels).^2) / sum((alphas_val - mean(alphas_val)).^2);

%%
function [X0, targets]  = create_mini_batch(X, yTrue, minibatch_size)

shuffle_idx = randperm(size(X, 3), minibatch_size);

X0 = X(:, :, shuffle_idx);
targets = yTrue(:, :, shuffle_idx);

if canUseGPU
    X0 = gpuArray(X0);
    targets = gpuArray(targets);
end

end


%%
function [mini_batch_inputs, targets] = call_batch(batch_inputs, batch_targets,iter,mbs)

mini_batch_inputs = batch_inputs(:, :, ((iter-1)*mbs)+1:(iter*mbs));
targets = batch_targets(:, :, ((iter-1)*mbs)+1:(iter*mbs));


end
