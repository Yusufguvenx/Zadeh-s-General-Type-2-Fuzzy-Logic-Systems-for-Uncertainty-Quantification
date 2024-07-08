% 20 JULY 2023
function output = T2_matrix_fuzzification_layer(x, membership_type,input_type, membership_speq, mf_count, input_count, mbs)
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

input_matrix = repmat(x, mf_count, 1, 1);
eps = 0;

output = zeros(mf_count, input_count, mbs, gpuArray);
output = custom_gaussmf(input_matrix, abs(membership_speq.pmf.input_sigmas), membership_speq.pmf.input_centers, input_count, eps);


end
%%

% Custom Gaussian function
function output = custom_gaussmf(x, s, c, n, eps)
    exponent = -0.5 * ((x - c).^2 ./ s.^2);
    exponent = 1/n*exponent;
    output = exp(exponent);

end