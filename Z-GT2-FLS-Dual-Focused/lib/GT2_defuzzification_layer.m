% 20 JULY 2023

function [output_lower, output_upper, output_mean] = GT2_defuzzification_layer(x,lower_firing_strength,upper_firing_strength, learnable_parameters,number_outputs, output_type,type_reduction_method,mbs,number_mf,u)
% v0.2 compatible with minibatch
% !!! not compatible with multiple outputs !!! will be written when needed
%
%
% calculating the weighted sum with firts calcutating the weighted
% elements then adding them
%
% @param output -> output
%
%       (1,1,mbs) tensor
%       mbs = mini-batch size
%       (:,:,1) -> defuzzified output of the first element of the batch
%
% @param input 1 -> normalized_firing_strength
%
%      (rc,1,mbs) tensor
%       rc = number of rules
%       mbs = mini-batch size
%       (1,1,1) -> normalized firing strength of the first rule of the
%       first element of the batch
%
% @param input 2 -> output_mf
%
%       (rc,1) vector
%       rc = number of rules
%       (1,1) -> constant or value of the first output membership function


        temp_mf = [learnable_parameters.pmf.linear.a,learnable_parameters.pmf.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

%         c = permute(c, [1, 3, 2]);

        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = pagemtimes(delta_f, u);
        
        pay2 = permute(c, [3, 1, 2]).*delta_f;
        pay2 = pagemtimes(pay2, u);
        pay2 = permute(pay2,[3,2,1]);

        pay1 = sum(c .* lower_firing_strength,1);
        pay1 = permute(pay1, [2, 1, 3]);

        pay = pay1 + pay2;

        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);
        payda1 = permute(payda1, [2, 1, 3]);

        payda = payda1 + payda2;

        output = pay./payda;

        output_lower = min(output,[],2);
        output_upper = max(output,[],2);
        output_mean = (output_lower + output_upper)./2;
%output = reshape(output, [s, b]);


output_lower = dlarray(output_lower);
output_upper = dlarray(output_upper);
output_mean = dlarray(output_mean);

end