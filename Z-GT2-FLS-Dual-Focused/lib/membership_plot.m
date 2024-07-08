function [x, y_lower, y_upper] = membership_plot(learnable_parameters, input_type, p, m)

x = -10:0.1:10;

if(input_type=="H")
    h = sigmoid(learnable_parameters.pmf.input_h(p, m));
    y_upper = custom_gaussmf(x, relu(learnable_parameters.pmf.input_sigmas(p, m)), learnable_parameters.pmf.input_centers(p, m));
    y_lower = h*y_upper;

elseif(input_type=="S")
    y_upper = custom_gaussmf(x, relu(learnable_parameters.pmf.input_sigmas(p, m)) + abs(learnable_parameters.pmf.delta_sigmas(p, m)), learnable_parameters.pmf.input_centers(p, m));
    y_lower = custom_gaussmf(x, relu(learnable_parameters.pmf.input_sigmas(p, m)) - abs(learnable_parameters.pmf.delta_sigmas(p, m)), learnable_parameters.pmf.input_centers(p, m));

elseif(input_type=="HS")
    h = sigmoid(learnable_parameters.pmf.input_h(p, m));
    y_upper = custom_gaussmf(x, relu(learnable_parameters.pmf.input_sigmas(p, m)) + abs(learnable_parameters.pmf.delta_sigmas(p, m)), learnable_parameters.pmf.input_centers(p, m));
    y_lower = custom_gaussmf(x, relu(learnable_parameters.pmf.input_sigmas(p, m)) - abs(learnable_parameters.pmf.delta_sigmas(p, m)), learnable_parameters.pmf.input_centers(p, m));
    y_lower = h*y_lower;
end

end

%% Custom Gaussian function
function output = custom_gaussmf(x, s, c)
    exponent = -0.5 * ((x - c).^2 ./ s.^2);
    output = exp(exponent);
end