% 22 JUN 2023
function [output_lower, output_upper] = T2_fuzzification_layer(x, membership_type,input_type, membership_speq, mf_count, input_count, mbs)
% v0.1 compatible with mini-batch
%
% calculating fuzzified values
%
% @param output -> output
%
%       (mfc,ic,mbs) tensor
%       mfc = number of input membership functions
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> fuzzified value of the first input with first membership
%       function of that input of first element of the batch
%
% @param input 1 -> x
%
%       (1,ic,mbs) tensor
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> firts input of the first element of the batch
%
% @param input 2 -> membership_type
%
%       a string
%       type of the membership function
%       it is gaussmf for now but gauss2mf will be added
%
% @param input 3 -> membership_speq
%
%       struct
%       consist of sigma and center values of each mf
%
% @param input 4 -> mf_count
%
%       constant
%       number of membership function for inputs
%
% @param input 5 -> input_count
%
%       constant
%       number of inputs
%
% @param input 6 -> mbs
%
%       constant
%       number of mini-batch size
%

output_upper = dlarray(zeros(mf_count, input_count, mbs));
output_lower = dlarray(zeros(mf_count, input_count, mbs));

% to ensure that h is between 0 and 1



if(membership_type == "gaussmf" && input_type == "H")

    membership_speq.input_h = sigmoid(membership_speq.input_h);
    
    for i = 1:input_count %number of inputs
        for j = 1:mf_count % number of mfs

            output_upper(j, i,:) = gaussmf(x(:,i,:), [membership_speq.input_sigmas(j, i) membership_speq.input_centers(j, i)]);


        end
    end


    output_lower = output_upper.*membership_speq.input_h;



elseif(membership_type == "gaussmf" && input_type == "S")

    for i = 1:input_count %number of inputs
        for j = 1:mf_count % number of mfs

            output_upper(j, i,:) = gaussmf(x(:,i,:), [(membership_speq.input_sigmas(j, i) + abs(membership_speq.delta_sigmas(j, i))) membership_speq.input_centers(j, i)]);
            output_lower(j, i,:) = gaussmf(x(:,i,:), [(membership_speq.input_sigmas(j, i) - abs(membership_speq.delta_sigmas(j, i))) membership_speq.input_centers(j, i)]);


        end
    end


elseif(membership_type == "gaussmf" && input_type == "HS")
    membership_speq.input_h = sigmoid(membership_speq.input_h);
    for i = 1:input_count %number of inputs
        for j = 1:mf_count % number of mfs

            output_upper(j, i,:) = gaussmf(x(:,i,:), [(membership_speq.input_sigmas(j, i) + abs(membership_speq.delta_sigmas(j, i))) membership_speq.input_centers(j, i)]);
            output_lower(j, i,:) = gaussmf(x(:,i,:), [(membership_speq.input_sigmas(j, i) - abs(membership_speq.delta_sigmas(j, i))) membership_speq.input_centers(j, i)]);


        end
    end

    output_lower = output_lower.*membership_speq.input_h;






elseif(membership_type ~= "gaussmf") %for future expansion
else %for future expansion
end

%output = dlarray(output);


end