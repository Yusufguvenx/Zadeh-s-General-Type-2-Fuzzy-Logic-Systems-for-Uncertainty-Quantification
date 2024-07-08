% Here, we are implementing gauss shape GT2 for boston housing dataset for
% an example. 
% Rule Number is 5.
% alpha = [0.01 0.5 1], (as we discussed in the last meeting)
% MAx-Min Normalization has been applied
% We use LA2 Learning Approach in this example
%Aiming %99 coverage with tau_upper = 0.995 and tau_lower = 0.005
%For LA2 loss definition, in GT2_fismodelLoss function We need to write
%GT2_fismodel_LA2_new (Important)

%%
clc;clear;
close all;
seed = 0;
rng(seed)
save_flag = "down"
%% Powerplant
dataset_name = "powerplant";
load("/home/yusuf/Desktop/fuzzy/Fuzzy-Code-Base-master/dataset/Datasets/CCPP.mat")

data = [x y];
training_num = 6697;
mbs = 512;
lr = 1e-2;

%%
current_path = pwd;
%where_to_save = "/home/yusuf/Desktop/fuzzy/tests_gauss_last_paper/tests_concrete";


number_mf = 5; % number of rules == number of membership functions

number_inputs = min(size(x));
number_outputs = min(size(y));

number_of_epoch = 100;

input_membership_type = "gaussmf";

input_type ="H";
% input_type ="S";
% input_type ="HS";

% output_membership_type = "singleton";
output_membership_type = "linear";
% output_membership_type = "IV";
% output_membership_type = "IVL";


% type_reduction_method = "SM";
% type_reduction_method = "BMM";
% type_reduction_method = "NT";
type_reduction_method = "KM";
% type_reduction_method ="Classical_KM"

delta = struct;
delta.delta1 = dlarray(zeros(1, 1));
delta.delta4 = dlarray(ones(1, 1));
alpha = [0.01 0.5 1];
alpha = permute(alpha, [4, 3, 1, 2]);
alpha_rev = (permute(alpha, [1, 4, 2, 3])) / sum(alpha); %to be used in LA1 and LA2

type_reduction_method_list = ["KM"];
% type_reduction_method_list = ["NT"];


gradDecay = 0.9;
sqGradDecay = 0.999;

PI_values = [];
test_results = [];

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

how_to_save = append("GT-2-",input_type,"-",output_membership_type,"-",type_reduction_method,"_gaussLA2","-", dataset_name,"-rng",string(seed))
%% Normalization upfront ------------------------------

%         [xn,input_mean,input_std] = zscore_norm(x);
%         [yn,output_mean,output_std] = zscore_norm(y);

[xn,input_min,input_range] = max_min_norm(x);
[yn,output_min,output_range] = max_min_norm(y);

data = [xn yn];
%% split by number ------------------------------

data_size = max(size(data));
test_num = data_size-training_num;

idx = randperm(data_size);

Training_temp = data(idx(1:training_num),:);
Testing_temp = data(idx(training_num+1:end),:);

%% ------------------------------

%training data
Train.inputs = reshape(Training_temp(:,1:number_inputs)', [1, number_inputs, training_num]); % traspose come from the working mechanism of the reshape, so it is a must
Train.outputs = reshape(Training_temp(:,(number_inputs+1:end))', [1, number_outputs, training_num]);

Train.inputs = dlarray(Train.inputs);
Train.outputs = dlarray(Train.outputs);

%testing data
Test.inputs = reshape(Testing_temp(:,1:number_inputs)', [1, number_inputs, test_num]);
Test.outputs = reshape(Testing_temp(:,(number_inputs+1:end))', [1, number_outputs, test_num]);


%% init

Learnable_parameters = initialize_Glorot_GT2(Train.inputs,input_type, Train.outputs,output_membership_type, number_mf);
prev_learnable_parameters = Learnable_parameters;


%% rng reset
rng(seed)

%% denormalizing for plotting

%         yTrue_train = max_min_denorm(reshape(Train.outputs,[number_outputs, training_num]),output_min,output_range);
%         yTrue_test = max_min_denorm(reshape(Test.outputs,[number_outputs, test_num]),output_min,output_range);

yTrue_train = reshape(Train.outputs, [1, max(size(Train.inputs))]);
yTrue_test = reshape(Test.outputs, [1, max(size(Test.inputs))]);
%%

number_of_iter_per_epoch = floorDiv(training_num, mbs);

number_of_iter = number_of_epoch * number_of_iter_per_epoch;
global_iteration = 1;

for epoch = 1: number_of_epoch

    [batch_inputs, batch_targets] = create_mini_batch(Train.inputs, Train.outputs, training_num);


    for iter = 1:number_of_iter_per_epoch

        [mini_batch_inputs, targets] = call_batch(batch_inputs, batch_targets,iter,mbs);

        [loss, gradients, yPred_train_lower, yPred_train_upper, yPred_train] = dlfeval(@GT2_fismodelLoss, mini_batch_inputs ,...
            number_inputs, targets,number_outputs, number_mf, mbs, Learnable_parameters, output_membership_type,...
            input_membership_type,input_type,type_reduction_method,u, alpha, delta, alpha_rev);
        
        [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(Learnable_parameters, gradients, averageGrad, averageSqGrad,...
            global_iteration, learnRate, gradDecay, sqGradDecay);



        global_iteration = global_iteration + 1;

    end

    %testing in each epoch
    [yPred_test_lower, yPred_test_upper, yPred_test] = GT2_fismodel_LA2_new(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha, delta, alpha_rev);
    
    yPred_test = reshape(yPred_test, [1, max(size(Test.inputs))]);
    yPred_test_upper = reshape(yPred_test_upper, [1, max(size(Test.inputs))]);
    yPred_test_lower = reshape(yPred_test_lower, [1, max(size(Test.inputs))]);
    


    iter_plot_T2(epoch,plotFrequency,loss,yTrue_test, yPred_test, yPred_test_upper, yPred_test_lower);

%             if (epoch == 70)
%                 learnRate = learnRate/10;
%             end


end
%% Inference
[yPred_train_lower, yPred_train_upper, yPred_train] = GT2_fismodel_LA2_new(Train.inputs, number_mf, number_inputs,number_outputs,length(Train.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u,alpha, delta, alpha_rev);
[yPred_test_lower, yPred_test_upper, yPred_test] = GT2_fismodel_LA2_new(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha, delta, alpha_rev);

yPred_train = reshape(yPred_train, [1, max(size(Train.inputs))]);
yPred_train_upper = reshape(yPred_train_upper, [1, max(size(Train.inputs))]);
yPred_train_lower = reshape(yPred_train_lower, [1, max(size(Train.inputs))]);

yPred_test = reshape(yPred_test, [1, max(size(Test.inputs))]);
yPred_test_upper = reshape(yPred_test_upper, [1, max(size(Test.inputs))]);
yPred_test_lower = reshape(yPred_test_lower, [1, max(size(Test.inputs))]);



train_RMSE = rmse(yPred_train, yTrue_train);
test_RMSE = rmse(yPred_test, yTrue_test);


PI_train = PICP(yTrue_train, yPred_train_lower, yPred_train_upper);
PI_test = PICP(yTrue_test, yPred_test_lower, yPred_test_upper);
PI_NAW_train = PINAW(yTrue_train, yPred_train_lower, yPred_train_upper);
PI_NAW_test = PINAW(yTrue_test, yPred_test_lower, yPred_test_upper);


PI_values = [PI_values, PI_test]
test_results = [test_results, test_RMSE]

%% ------------------------------
if save_flag == "up"
    cd(where_to_save)

    if ~exist(how_to_save, 'dir')
        mkdir(how_to_save);
    end

    cd(how_to_save)
    savefig
    save
    cd(current_path)
end

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
