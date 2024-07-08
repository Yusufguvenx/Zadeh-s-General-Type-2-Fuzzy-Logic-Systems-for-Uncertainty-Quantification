function learnable_parameters = initialize_Glorot_GT2(input_data, input_type, output_data, output_type, number_mf)

%% centers with Kmean
number_inputs = size(input_data,2);
number_outputs = size(output_data,2);

data = [input_data output_data];
% data = extractdata(reshape(data,[size(data,3) size(data,2)]));
% data = extractdata(permute(data, [3, 2, 1]));
% [~,centers] = kmeans(data,number_mf);
% learnable_parameters.pmf.input_centers = centers(:,1:end-number_outputs);

input_data = extractdata(permute(input_data, [3 2 1]));

for i = 1:size(input_data, 2)

    [~, centers(:, i)] = kmeans(input_data(:, i), number_mf, "MaxIter",100);
end

% [~, centers] = kmeans(input_data, number_mf);


learnable_parameters.pmf.input_centers = dlarray(centers);

% learnable_parameters.pmf.input_centers = dlarray(learnable_parameters.pmf.input_centers);

%% sigmas

s = std(input_data); 
s(s == 0) = 1;
% s = 1*std(input_data)./std(input_data); 


s = repmat(s,number_mf,1);

learnable_parameters.pmf.input_sigmas = s;
learnable_parameters.pmf.input_sigmas = dlarray(learnable_parameters.pmf.input_sigmas);

if input_type == "H"

    h = rand(number_mf,number_inputs);

    learnable_parameters.pmf.input_h = h;

    learnable_parameters.pmf.input_h = dlarray(learnable_parameters.pmf.input_h);

elseif input_type == "S"

    delta_sigma = rand(number_mf,number_inputs)*0.01;

    learnable_parameters.pmf.delta_sigmas = delta_sigma;

    learnable_parameters.pmf.delta_sigmas = dlarray(learnable_parameters.pmf.delta_sigmas);

elseif input_type == "HS"

    h = rand(number_mf,number_inputs);

    learnable_parameters.pmf.input_h = h;

    learnable_parameters.pmf.input_h = dlarray(learnable_parameters.pmf.input_h);

    delta_sigmas = rand(number_mf,number_inputs)*0.01;

%     delta_sigmas = rand(number_mf,number_inputs);
    learnable_parameters.pmf.delta_sigmas = delta_sigmas;

    learnable_parameters.pmf.delta_sigmas = dlarray(learnable_parameters.pmf.delta_sigmas);

end

%%

if output_type == "singleton"

    c = rand(number_mf,number_outputs)*0.01;
    learnable_parameters.pmf.singleton.c = dlarray(c);

elseif output_type == "linear"

    a = rand(number_mf*number_outputs,number_inputs)*0.01;
    learnable_parameters.pmf.linear.a = dlarray(a);


    b = rand(number_mf*number_outputs,1)*0.01; % single output
    learnable_parameters.pmf.linear.b = dlarray(b);


elseif output_type == "IV"

    c = rand(number_mf,number_outputs)*0.01;

    delta = rand(number_mf,number_outputs)*0.01;

    learnable_parameters.pmf.IV.c = dlarray(c);
    learnable_parameters.pmf.IV.delta = dlarray(delta);




elseif output_type == "IVL"

    a = rand(number_mf*number_outputs,number_inputs)*0.01;
    b = rand(number_mf*number_outputs,1)*0.01; % single output


    delta_a = rand(number_mf*number_outputs,number_inputs)*0.01;
    delta_b = rand(number_mf*number_outputs,1)*0.01; % single output

%     a = rand(number_mf*number_outputs,number_inputs);
%     b = rand(number_mf*number_outputs,1); % single output
% 
% 
%     delta_a = rand(number_mf*number_outputs,number_inputs);
%     delta_b = rand(number_mf*number_outputs,1); % single output

    learnable_parameters.pmf.IVL.a = dlarray(a);
    learnable_parameters.pmf.IVL.delta_a = dlarray(delta_a);
    learnable_parameters.pmf.IVL.b = dlarray(b);
    learnable_parameters.pmf.IVL.delta_b = dlarray(delta_b);


end

% %GT2 learnable parameters

% delta2 = 0.5;
% theta2 = 0.5;
% 
% learnable_parameters.smf.delta2 = dlarray(delta2);
% learnable_parameters.smf.theta2 = dlarray(theta2);

% Batu Version

% delta3 = 0.5;
% delta2 = 0.5;
% 
% learnable_parameters.smf.delta2 = dlarray(delta2);
% learnable_parameters.smf.delta3 = dlarray(delta3);


lower_sigma = rand(1, 1) * 0.1;
upper_sigma = rand(1, 1) * 0.1;

learnable_parameters.smf.lower_sigma = dlarray(lower_sigma);
learnable_parameters.smf.upper_sigma = dlarray(upper_sigma);




end