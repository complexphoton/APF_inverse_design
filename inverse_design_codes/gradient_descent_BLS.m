function [y_edge_list, FoM_list, kkk] = gradient_descent_BLS(y_edge_list, h, syst, sim_info, channels, other_derivatives, BLS_parameter, options)

% GRADIENT_DESCENT_BLS performs the gradient-descent optimization with the learning rate determined by
% the backtracking line search for a metasurface beam splitter optimization
% problem.

%   === Input Arguments ===
%   y_edge_list (numeric row vector; required):                       
%         y coordinates of edges of ridges (optimization variables) [micron]
%   h (numeric scalar; required):                                      
%         Metasurface thickness [micron]
%   syst (scalar structure; required):
%          A structure that specifies the system, used to build the FDFD matrix A in mesti.
%          See the helping documents of mesti for more details.
%   sim_info (scalar structure; required):
%          A structure that provides the refractive indices of the system,
%          and pixel numbers in each direction of the system. It
%          contains the following fields (all numeric scalars):
%          sim_info.RI:  Refractive index
%               sim_info.RI.n_bg:           Background refractive index
%               sim_info.RI.n_sub:         Refractive index of the substrate
%               sim_info.RI.n_ridge:       Refractive index of the ridge
%          sim_info.num_pixels:  Pixel numbers in each direction
%               sim_info.num_pixel.nz:                   Pixels of metasurfaces in the z direction
%               sim_info.num_pixel.ny:                   Pixels of metasurfaces in the y direction
%               sim_info.num_pixel.nz_extra_left:   Pixels added on the left side in the z direction
%               sim_info.num_pixel.nz_extra_right: Pixels added on the right side in the z direction
%               sim_info.num_pixel.ny_extra_high:  Pixels added on the upper side in the y direction
%               sim_info.num_pixel.ny_extra_low:    Pixels added on the lower side in the y direction
%   channels (structure array; required)
%          A structure that provides information of channels on the
%          left/right of the system. See the helping documents of mesti for more details.  
%          It contains the following fields:
%          channels.L.B_L (2D matrix):                              Source profiles on the left
%          channels.L.N_L (numeric scalar):                       Number of channels on the left
%          channels.L.pos (four-element integer vector):  Position of block source on the left
%          channels.R.B_R (2D matrix):                             Source profiles on the right
%          channels.R.N_R (numeric scalar):                     Number of channels on the right
%          channels.R.pos (four-element integer vector): Position of block source on the left
%          channels.sqrt_nu_L (1-by-N_L row vector):     Flux-normalization prefactor sqrt(nu) on the left
%          channels.sqrt_nu_R (1-by-N_R row vector):    Flux-normalization prefactor sqrt(nu) on the right
%   other_derivatives (structure array; required)
%          A structure that contains the function handles of the figure of merit (FoM), \partial f/
%          \partial pk, and \partial f/\partial T. It contains the following fields:
%          other_derivatives.FOM:     Function handle of the FoM
%          other_derivatives.pfppk:   Function handle of \partial f/\partial pk
%          other_derivatives.dfdT:     Function handle of \partial f/\partial T
%   BLS_parameter (scalar structure; required)
%          A structure that contains parameters needed for the backtracking line search. It contains the following fields (all numeric scalars):
%          BLS_parameter.min_feature:                          Minimal feature size for fabrication consideration [micron]
%          BLS_parameter.gamma_max:                         Maximum learning rate [micron^2]
%          BLS_parameter.tau:                                        Shrink factor of the learning rate (<1)
%          BLS_parameter.c1:                                         c1 in the Armijo condition
%          BLS_parameter.constraint:                             Constraints for the optimization
%              BLS_parameter.constraint.y_low:                Lower bound of edge positions [micron]
%              BLS_parameter.constraint.y_high:              Upper bound of edge positions [micron]
%              BLS_parameter.constraint.num_eval_max: Maximal iteration number
%              BLS_parameter.constraint.ftol_abs:            Optimization stops when |f(P_{n+1})-f(P_n)| < ftol_abs
%   options (scalar structure; optional, defaults to an empty struct):                               
%          A structure that specifies the options of computation; defaults to an
%          empty structure. It can contain the following fields:
%          options.symmetry (logical scalar, defaults to false):
%                Whether or not a metasurface is symmetric with respect to its central plane.
%          options.method (character vector, defalts to APF): 
%                The solution method. Available choices are (case-insensitive):
%                'APF':       Augmented partial factorization
%                'adjoint':  Adjoint method
%          options.gradient (logical scalar, defaults to true):
%                Whether or not to compute the gradient of the FoM
%          options.N_sub (numeric scalar, defaults to 1):
%                Divide one large APF computation yielding both the transmission matrix and 
%                the gradient into N_sub small APF computations to save time and memory.
%                N_sub must be an integer scalar > 0.

%   === Output Arguments ===
%   y_edge_list (numeric row vector):                       
%          Optimized y coordinates of ridge edges [micron]
%   FoM_list (numeric column vector):                           
%          FoM of each iteration
%   kkk (numeric scalar):
%          Number of iterations costed by the optimization

if nargin < 7
    error('Not enough input arguments.');
end

% Extract BLS parameters
min_feature = BLS_parameter.min_feature;                           % Minimal feature size [micron]
gamma_max = BLS_parameter.gamma_max;                          % Maximum learning rate [micron^2]
tau = BLS_parameter.tau;                                                     % Shrink factor of the learning rate (<1)
c1 = BLS_parameter.c1;                                                       % c1 in the Armijo condition
y_low = BLS_parameter.constraint.y_low;                              % Lower bound of edge positions [micron]
y_high = BLS_parameter.constraint.y_high;                           % Upper bound of edge positions [micron] 
num_eval_max = BLS_parameter.constraint.num_eval_max;  % Maximal number of iteration
ftol_abs = BLS_parameter.constraint.ftol_abs;                       % Optimization stops when |f(P_{n+1})-f(P_n)| < ftol_abs 

% Number of edges (ie, optimization variables)
num_edge_change = length(y_edge_list);   

% Store the FoM as the optimization goes on
FoM_list = zeros(num_eval_max,1);

% Store the learning rate used in each iteration
gamma_list = zeros(num_eval_max,1);

% Run optimization
for kkk = 1:num_eval_max
    
    % Constrain optimization variables
    % Bound constraint: [y_low, y_high]
    if y_edge_list(end) > y_high
        y_edge_list(end) = y_high;
    elseif y_edge_list(1) < y_low
        y_edge_list(1) = y_low;
    end
    % Inequality constraint: y_{ii+1} - y_ii >= min_feature
    for ii = 1:num_edge_change
        if ii > 1
           y_distance = y_edge_list(ii) - y_edge_list(ii-1);
           if y_distance < min_feature
               y_add = (min_feature - y_distance)/2;
               y_edge_list(ii-1) = y_edge_list(ii-1) - y_add;
               y_edge_list(ii) = y_edge_list(ii) + y_add;
           end
        end
    end
    
    options.gradient = true;
    [FoM_val, FoM_grad] = FoM_and_grad(y_edge_list, h, syst, sim_info, channels, other_derivatives, options);
    
    % Store FoM in each iteration
    FoM_list(kkk) = FoM_val;
    
     % Check if |f(P_{n+1})-f(P_n)| < ftol_abs has satisfied.
     % If it has, end the optimizatipn
    if kkk ~= 1 && abs(FoM_list(kkk) - FoM_list(kkk-1)) < ftol_abs
        break
    end

    % Update the optimization variables
    % Use backtracking line search with Armijo criterion to decide the learning rate gamma_n
    % Choose the maximal gamma_n for each iteration to save time.
    % Divide gamma by tau here to compensate the multiplication of tau when
    % we first enter the following loop.
    if kkk > 1
        gamma = min(gamma_max, 10*gamma_list(kkk-1))/tau;
    else
        gamma = gamma_max/tau;
    end
    
    % Check if the Armijo condition is satisfied.
    % Yes => use the current gamma_n to update the optimization variables
    % No  => shink gamma_n by tau, and repeat this checking process
    FOM_bls = FoM_val;    % Initial value of the left-hand side of the Armijo criterion, making sure that we enter the loop at first
    while FOM_bls > FoM_val - c1*gamma*FoM_grad*FoM_grad.'
        
        gamma = tau*gamma;
        
        % P_n - gamma_n*dfdP
        y_edge_list_bls = y_edge_list-gamma*FoM_grad;
        % Constrain optimization variables
        % Bound constraint: [y_low, y_high]
        if y_edge_list_bls(end) > y_high
             y_edge_list_bls(end) = y_high;
        elseif y_edge_list_bls(1) < y_low
             y_edge_list_bls(1) = y_low;
        end
        % Inequality constraint: y_{ii+1} - y_ii >= min_feature
        for ii = 1:num_edge_change
            if ii > 1
               y_distance = y_edge_list_bls(ii) - y_edge_list_bls(ii-1);
               if y_distance < min_feature
                   y_add = (min_feature - y_distance)/2;
                   y_edge_list_bls(ii-1) = y_edge_list_bls(ii-1) - y_add;
                   y_edge_list_bls(ii) = y_edge_list_bls(ii) + y_add;
               end
            end
        end
        
        options.gradient = false;
        FOM_bls = FoM_and_grad(y_edge_list_bls, h, syst, sim_info, channels, other_derivatives, options);
        
    end
    
    % Store the gamma_n used in each iteration
    gamma_list(kkk) = gamma;
    
    % Update the edge positions
    y_edge_list = y_edge_list  - gamma*FoM_grad;
 
end

end