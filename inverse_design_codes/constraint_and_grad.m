function [val, grad] = constraint_and_grad(y_edge_list, min_feature, kkk)

% CONSTRAINT_AND_GRAD builds the inequality constraint and its gradient for fabrication requirements
% y_{k+1} - y_k = dy >= min_feature
% equivalent to: y_k - y_{k+1} + min_feature <= 0

%   === Input Arguments ===
%   y_edge_list (numeric row vector; required):                       
%         y coordinates of edges of ridges (optimization variables) [micron]
%   min_feature (numeric scalar; required):         
%         Minimal feature size [micron]
%   kkk (numeric scalar; required):                      
%         Index of the optimization variable

%   === Output Arguments ===
%   val (numeric scalar):                       
%        The value of y_k - y_{k+1} + min_feature
%   grad (numeric row vector):                    
%        The gradient of y_k - y_{k+1} + min_feature


if nargin < 3
    error('Not enough input arguments.');
end

ny_edge = length(y_edge_list);

% Note that the fabrication constraints on the first and last optimization variables
% have been taken care of at the bound constraints, so we only need to
% consider the distance between neighboring edge positions here.
grad = zeros(1, ny_edge);
if kkk < ny_edge
    val = y_edge_list(kkk) - y_edge_list(kkk+1) + min_feature;
    grad(kkk) = 1;
    grad(kkk+1) = -1;
else
    val = 0;
end

end