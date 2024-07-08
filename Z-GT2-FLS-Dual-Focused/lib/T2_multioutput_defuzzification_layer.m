% 01 AUG 2023

function [output_lower, output_upper, output_mean] = T2_multioutput_defuzzification_layer(x,lower_firing_strength,upper_firing_strength, learnable_parameters,number_outputs, output_type,type_reduction_method,mbs,number_mf,u)
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
%       (rc,1,mbs) tensor
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

if output_type == "singleton"

    if type_reduction_method == "SM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);

        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);

        output_lower = normalized_lower_firing_strength.* learnable_parameters.singleton.c;% first we multiply elementwise our firing strengths with output memberships
        output_upper = normalized_upper_firing_strength.* learnable_parameters.singleton.c;

        output_lower = sum(output_lower, 1); %then we sum with respect to dimension 1
        output_upper = sum(output_upper, 1); %then we sum with respect to dimension 1

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NT"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);


        numerator_lower = lower_firing_strength_temp.* learnable_parameters.singleton.c;
        numerator_upper = upper_firing_strength_temp.* learnable_parameters.singleton.c;

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength, 1) + sum(upper_firing_strength, 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "KM"

        
        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

        pay2 = reshape((permute(learnable_parameters.singleton.c,[3 2 1]).*permute(delta_f,[1 3 2])),[mbs*number_outputs,number_mf])*u;
        pay2 = reshape(pay2,mbs, number_outputs,[]);
        pay2 = permute(pay2,[3,2,1]);
        pay1 = sum(learnable_parameters.singleton.c.* lower_firing_strength,1);

        pay = pay1 + pay2;

        %         clear pay1_upper pay2_upper
        %         clear delta_f u

        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;

        %         clear payda1 payda2

        pay = permute(pay,[2 1 3]);

        output = pay./payda;
        
        %         clear pay_lower pay_upper payda

        output_lower = min(output,[],2);
        output_upper = max(output,[],2);

        %         clear output_lower_temp output_upper_temp

        output_mean = (output_lower + output_upper)./2;

    end

elseif output_type == "linear"


    if type_reduction_method == "SM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        c = temp_mf*temp_input;
        c= reshape(c, [size(normalized_lower_firing_strength, 1), number_outputs, size(normalized_lower_firing_strength, 3)]);


        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);


        output_lower = normalized_lower_firing_strength.* c;
        output_upper = normalized_upper_firing_strength.* c;

        output_lower = sum(output_lower, 1); %then we sum with respect to dimension 1
        output_upper = sum(output_upper, 1); %then we sum with respect to dimension 1

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "NT"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        numerator_lower = lower_firing_strength_temp.* c;
        numerator_upper = upper_firing_strength_temp.* c;

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength, 1) + sum(upper_firing_strength, 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "KM"

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

        pay2 = (permute(c,[3 1 2]).*delta_f)*u;
        pay2 = permute(pay2,[3,2,1]);
        pay1 = sum(c .* lower_firing_strength,1);

        pay = pay1 + pay2;

        %         clear pay1_upper pay2_upper
        %         clear delta_f u

        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;

        %         clear payda1 payda2

        output = pay./payda;
        
        %         clear pay_lower pay_upper payda

        output_lower = min(output,[],2);
        output_upper = max(output,[],2);

        %         clear output_lower_temp output_upper_temp

        output_mean = (output_lower + output_upper)./2;

    end





elseif output_type == "IV"

    if type_reduction_method == "SM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);


        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);


        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);


        output_lower = normalized_lower_firing_strength.* c_lower;
        output_upper = normalized_upper_firing_strength.* c_upper;

        output_lower = sum(output_lower, 1);
        output_upper = sum(output_upper, 1);

        output_mean = (output_lower + output_upper)./2;



    elseif type_reduction_method == "NT"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);


        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);


        numerator_lower = lower_firing_strength_temp.* c_upper;
        numerator_upper = upper_firing_strength_temp.* c_lower;

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength, 1) + sum(upper_firing_strength, 1);


        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "KM"


        c_upper = learnable_parameters.IV.c + abs(learnable_parameters.IV.delta);
        c_lower = learnable_parameters.IV.c - abs(learnable_parameters.IV.delta);


        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

        pay2_lower = (permute(c_lower,[3 1 2]).*delta_f)*u;
        pay2_lower = permute(pay2_lower,[3,2,1]);
        pay1_lower = sum(c_lower .* lower_firing_strength,1);

        pay_lower = pay1_lower + pay2_lower;



        pay2_upper = (permute(c_upper,[3 1 2]).*delta_f)*u;
        pay2_upper = permute(pay2_upper,[3,2,1]);
        pay1_upper = sum(c_upper .* lower_firing_strength,1);

        pay_upper = pay1_upper + pay2_upper;


        %         clear pay1_upper pay2_upper
        %         clear delta_f u


        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;

        %         clear payda1 payda2


        output_lower = pay_lower./payda;
        output_upper = pay_upper./payda;

        %         clear pay_lower pay_upper payda

        output_lower = min(output_lower,[],2);
        output_upper = max(output_upper,[],2);

        %         clear output_lower_temp output_upper_temp

        output_mean = (output_lower + output_upper)./2;

    end



elseif output_type == "IVL"


    if type_reduction_method == "SM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);


        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];


        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(normalized_lower_firing_strength, 1), number_outputs, size(normalized_lower_firing_strength, 3)]);
        linear_upper = reshape(linear_upper, [size(normalized_upper_firing_strength, 1), number_outputs, size(normalized_upper_firing_strength, 3)]);

        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);

        output_lower = normalized_lower_firing_strength.* linear_lower;
        output_upper = normalized_upper_firing_strength.* linear_upper;

        output_lower = sum(output_lower, 1);
        output_upper = sum(output_upper, 1);

        output_mean = (output_lower + output_upper)./2;


    elseif type_reduction_method == "NT"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        numerator_lower = lower_firing_strength_temp.* linear_lower;
        numerator_upper = upper_firing_strength_temp.* linear_upper;

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength, 1) + sum(upper_firing_strength, 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;

    elseif type_reduction_method == "KM"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.IVL.a,learnable_parameters.IVL.b];
        temp_delta = [learnable_parameters.IVL.delta_a,learnable_parameters.IVL.delta_b];

        x = permute(x,[2 1 3]); %comment at
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        linear_lower = (temp_mf * temp_input) - (abs(temp_delta * temp_input));
        linear_upper = (temp_mf * temp_input) + (abs(temp_delta * temp_input));

        linear_lower = reshape(linear_lower, [size(lower_firing_strength_temp, 1), number_outputs, size(lower_firing_strength_temp, 3)]);
        linear_upper = reshape(linear_upper, [size(upper_firing_strength_temp, 1), number_outputs, size(upper_firing_strength_temp, 3)]);

        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

        pay2_lower = (permute(linear_lower,[3 1 2]).*delta_f)*u;
        pay2_lower = permute(pay2_lower,[3,2,1]);
        pay1_lower = sum(linear_lower .* lower_firing_strength,1);

        pay_lower = pay1_lower + pay2_lower;

        pay2_upper = (permute(linear_upper,[3 1 2]).*delta_f)*u;
        pay2_upper = permute(pay2_upper,[3,2,1]);
        pay1_upper = sum(linear_upper .* lower_firing_strength,1);

        pay_upper = pay1_upper + pay2_upper;
        %         clear pay1_upper pay2_upper
        %         clear delta_f u
        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;
        %         clear payda1 payda2
        output_lower = pay_lower./payda;
        output_upper = pay_upper./payda;
        %         clear pay_lower pay_upper payda
        output_lower = min(output_lower,[],2);
        output_upper = max(output_upper,[],2);
        %         clear output_lower_temp output_upper_temp
        output_mean = (output_lower + output_upper)./2;

    end


end
%output = reshape(output, [s, b]);


output_lower = dlarray(output_lower);
output_upper = dlarray(output_upper);
output_mean = dlarray(output_mean);

end