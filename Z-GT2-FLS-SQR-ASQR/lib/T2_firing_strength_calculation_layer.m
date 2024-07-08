% 15 JULY 2023
function [output_lower, output_upper] = T2_firing_strength_calculation_layer(lower_membership_values,upper_membership_values,operator_type)
% v0.2 compatibel with mini-batch
% more operators can be added
%
% rule inferance for static rules, rule count is equal to number of input membership
% function or number of outputs
% 
% @param output -> output
%
%       (n,1,mb) vector 
%       n = number of rows in input
%       (:,1,1) -> firing strength of each rule
%
% @param input 1 -> membership_values
%
%       (n,m) vector 
%       n = number of input membership
%       function or number of outputs
%       m = number of inputs to FIS system or number of features
%       (:,1) -> fuzzified values of input one for each membership function
%       (1,:) -> fuzzified output of membership fuction 1 of each input
%
% @param input 2 -> operator_type
%
%       a string
%       operator type that will be applied in rules
%       4 options are available for now: "product" , "sum" , "max" , "min"
%       
if operator_type == "product"

    lower_firing_strength = prod(lower_membership_values,2);
    upper_firing_strength = prod(upper_membership_values,2);




elseif operator_type == "sum"

    lower_firing_strength = sum(lower_membership_values,2);
    upper_firing_strength = sum(upper_membership_values,2);

elseif operator_type == "max"

    lower_firing_strength = max(lower_membership_values,[],2);
    upper_firing_strength = max(upper_membership_values,[],2);

elseif operator_type == "min"

    lower_firing_strength = min(lower_membership_values,[],2);
    upper_firing_strength = min(upper_membership_values,[],2);

elseif operator_type == "deneme"

    lower_firing_strength = 1 - mean((1 - lower_membership_values),2);
    upper_firing_strength = 1 - mean((1 - upper_membership_values),2);

end

output_lower = lower_firing_strength;
output_upper = upper_firing_strength;

end