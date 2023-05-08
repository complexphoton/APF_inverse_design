%% Inverse design of a metasurface beam splitter with a broad incident angular range

% Consider an infinite-sized, periodic metasurface with N = 80 meta-atoms
% each period.
% Periodic boundary condition in the y direction, perfectly matched layers
% (PMLs) in the z direction.

%% Prepare for the inverse design

% Set the seed of random number generator to generate different initial guesses for reproducibility
rand_seed = 1;

% Whether the metasurface is symmetric with respect to its central plane (default to false)
options.symmetry = true;
% Use APF method for the objective-plus-gradient computation
% Another option is the adjoint method: options.method = 'adjoint'
% default to APF
options.method = 'APF';
% Speed up the APF gradient computation by matrix division (Sec.2.2 of the inverse design paper)
% Dividing one large APF computation into N_sub small APF computations, and
% combine their results together
% If N_sub = 1, no matrix division
options.N_sub = 3;

% Whether or not to use the open-source package NLopt:
% https://github.com/stevengj/nlopt to do the optimization
% If NLopt is disabled, the optimization will be performed using
% the gradient-descent method with backtracking line search (by default), or the
% function fmincon in Matlab if specified.
using_NLopt = true;

%%  Design parameters

t1_tot = clock;

n_air    = 1.0;                 % Refractive index of air on the right
n_silica = 1.45;              % Refractive index of silica substrate on the left
n_aSi   = 3.70;                % Refractive index of a-Si ridges
wavelength = 0.94;         % Vacuum wavelength [micron]
dx = wavelength/40;      % Discretization grid size [micron]
FOV = 60;                       % Angular bandwidth of the metasurface in the air [deg]

% Thickness of the metasurface [micron],
% which is chosen to provide 2pi phase shift with high transmission by
% changing the width of each ridge [see Supplementary Fig.S3 of the inverse design paper].
h  = 0.56;   

% Store refractive indices and use them in the function FOM_AND_GRAD
sim_info.RI.n_bg = n_air;
sim_info.RI.n_sub = n_silica;
sim_info.RI.n_ridge = n_aSi;

n_meta_atom = 80;         % Number of meta atoms
% Approximate width of the metasurface [micron]
% As shown in Supplementary Fig.S3, the periodicity of each unit cell is about 0.3*wavelength.
% Note that we use the periodicity of unit cell to estimate the width of
% the metasurface, but we don't limit the metasurface to conventional unit-cell-based designs.
W  = ceil(n_meta_atom*wavelength*0.3/2)*2;          

ny = ceil(W/dx);      % Number of pixels of the metasurface in the y direction
dx = W/ny;             % Update dx to fit W into integer number of pixels 
nz = ceil(h/dx);      % Number of pixels of the metasurface in the z direction

nPML = 20;     % Number of pixels of PMLs
% PMLs on the z direction, add one pixel of free space for source and projection
nz_extra_left = 1 + nPML;
nz_extra_right = nz_extra_left;
% Periodic BC on the y direction
ny_extra_low = 0;
ny_extra_high = ny_extra_low;
% Number of pixels for the whole system with PMLs
ny_tot = ny + ny_extra_low + ny_extra_high;
% Store pixel numbers and use them in the function FOM_AND_GRAD
sim_info.num_pixel.nz = nz;
sim_info.num_pixel.ny = ny;
sim_info.num_pixel.nz_extra_left = nz_extra_left;
sim_info.num_pixel.nz_extra_right = nz_extra_right;
sim_info.num_pixel.ny_extra_low = ny_extra_low;
sim_info.num_pixel.ny_extra_high = ny_extra_high;

k0dx = 2*pi/wavelength*dx;  % Dimensionless frequency k0*dx
epsilon_L = n_silica^2;           % Relative permittivity on the left
epsilon_R = n_air^2;               % Relative permittivity on the right and on the top & bottom

%% Build input profile B
    
% Obtain properties of propagating channels on the two sides.
BC = 'periodic'; % Periodic boundary condition means the propagating channels are plane waves
use_continuous_dispersion = true; % Use continuous dispersion relation for (kx,ky)
channels_L = mesti_build_channels(ny, 'TM', BC, k0dx, epsilon_L, [], use_continuous_dispersion);
channels_R = mesti_build_channels(ny, 'TM', BC, k0dx, epsilon_R, [], use_continuous_dispersion);

% We use all propagating plane-wave channels on the right
N_R = channels_R.N_prop;    % Number of channels on the right

% For the incident plane waves, we only take channels within the desired incident angular range:
% |ky| < n_air*k0*sin(FOV/2)
kydx_bound = n_air*k0dx*sind(FOV/2);
ind_kydx_FOV = find(abs(channels_L.kydx_prop) < kydx_bound);
N_L = numel(ind_kydx_FOV);                                    % Number of channels on the left
kydx_FOV = channels_L.kydx_prop(ind_kydx_FOV);  % kydx within the angular bandwidth of interest

M = N_L + N_R;   % Get the total number of input (output) channels coming from both two sides of the metasurface

% Note that even though we only need the transmission matrix with input
% from the left, here we also include input channels from the right (same
% as the output channels of interest). This allows us to make matrix K =
% [A,B;C,0] symmetric.
B_L = channels_L.fun_u(kydx_FOV);                        % Build the input matrix on the left surface
B_R = channels_R.fun_u(channels_R.kydx_prop);    % Build the input matrix on the right surface

% In mesti(), B_struct.pos = [m1, n1, h, w] specifies the position of a
% block source, where (m1, n1) is the index of the smaller-(y,x) corner,
% and (h, w) is the height and width of the block. Here, we put line
% sources (w=1) on the left surface (n1=n_L) and the right surface
% (n1=n_R) with height ny centered around the metalens.
n_L = nz_extra_left;               % x pixel immediately before the metalens
n_R = n_L + nz + 1;               % x pixel immediately after the metalens
m1_L = ny_extra_low +1;      % first y pixel of the metalens
m1_R = m1_L;                        % first y pixel of the output projection window

% Store information of channels and use them in the function FOM_AND_GRAD
channels.L.B_L = B_L;
channels.L.N_L = N_L;
channels.L.pos =  [m1_L, n_L, ny, 1];
channels.R.B_R = B_R;
channels.R.N_R = N_R;
channels.R.pos =  [m1_R, n_R, ny, 1];
% The flux-normalization prefactor sqrt(nu) on the left/right
channels.sqrt_nu_L = channels_L.sqrt_nu_prop(ind_kydx_FOV);
channels.sqrt_nu_R = channels_R.sqrt_nu_prop;

% Simulation domain information; will use in the function FOM_AND_GRAD
syst.length_unit = 'µm';
syst.wavelength = wavelength;
syst.dx = dx;
syst.yBC = 'periodic';          % Periodic BC in the y direction
syst.PML.npixels = nPML;   % Number of PML pixels
syst.PML.direction = 'x';     % Only place PML in the x direction

%% Inverse design of a metasurface beam splitter

 % Build the target transmission matrix,
 % with T_nm = 0.5 at the plus and minus 1 diffraction orders and T_nm = 0 otherwise. 
 T_target = zeros(N_R, N_L);
 ind_target= round(interp1(channels_R.kydx_prop, 1:N_R, kydx_FOV, 'linear', 'extrap'));
 for ii = 1:N_L
     T_target(ind_target(ii)+[-1,1], ii) = 1/2;
 end

 % Function handles of the figure of merit (FoM) [Eq.(6) of the inverse
 % design paper], \partial f/\partial pk and \partial f/\partial T in
 % Eq.(1) of the inverse design paper
 other_derivatives.FOM = @(x) sum(reshape((abs(x).^2 - T_target).^2, [], 1));
 other_derivatives.pfppk = @(x) 0;
 other_derivatives.dfdT = @(x) 2*(abs(x).^2 - T_target).*conj(x); 

% Generate initial guess
min_feature = 0.04;                     % Minimal feature size = 0.04 [micron] for fabrication consideration
y_low = -W/2 + min_feature/2;   % Lower bound of edge positions [micron]
% If the metasurface is symmetric with respect to its central plane
if options.symmetry    
    % For symmetric metasurfaces, we parameterize the two edge positions of the
    % n_meta_atom/2 ridges on the left-half side of the metasurface, and get
    % the right-half side based on symmetry.
    num_edge_change = n_meta_atom;     % Number of optimization variables (ie, edges of ridges)
    
    y_high = 0 - min_feature/2;        % Upper bound of edge positions [micron]
    extra_space = W/2 - min_feature - min_feature*(num_edge_change-1);
% General metasurface without mirror symmetry with respect to its central plane
else   
    % For general metasurfaces, we parameterize the two edge positions of all
    % n_meta_atom ridges of the metasurface,
    num_edge_change = 2*n_meta_atom;         % Number of optimization variables (ie, edges of ridges)
    
    y_high = W/2 - min_feature/2;   % Upper bound of edge positions [micron]
    extra_space = W - min_feature - min_feature*(num_edge_change-1);
end
% Generate the initial guess with all features larger than min_feature
rng(rand_seed)
y_temp = sort(extra_space*rand(1,num_edge_change) + y_low, 'ascend');
y_edge_list_init = (0:min_feature:min_feature*(num_edge_change-1)) + y_temp;

% Stop criteria
num_eval_max = 3000;     % Maximal number of iteration
ftol_abs = 1e-4;          % Optimization stops when |f(P_{n+1})-f(P_n)| < ftol_abs

% Choose the optimization method
if using_NLopt
    % To use Nlopt, we first need to add its path to Matlab.
    % See more details on the website: https://github.com/stevengj/nlopt
    
    % Choose the local gradient-based algorithm: SLSQP (index: 40) or MMA (index:24), supporting inequality constraints
    ind_algrithm = 40;

    % Set up Nlopt
    opt.algorithm = ind_algrithm;   % Choose the algorithm
    opt.maxeval = num_eval_max;  % Maximal iteration number
    opt.ftol_abs = ftol_abs;             % Optimization stops when |f(P_{n+1})-f(P_n)| < ftol_abs
    opt.verbose = 1;                        % Output FoM of each iteration 
    % The lower and upper bounds of each optimization variable
    opt.lower_bounds = y_low:min_feature:(y_low+min_feature*(num_edge_change-1));
    opt.upper_bounds = (y_high-min_feature*(num_edge_change-1)):min_feature:y_high;
    % The inequality constraints: y_{k+1} - y_k >= min_feature
    opt.fc = cell(1,num_edge_change);
    for kkk = 1:num_edge_change
       opt.fc{kkk} = @(y_edge_list)  constraint_and_grad(y_edge_list, min_feature, kkk);
    end

    % The function used to compute the FoM and gradient in each iteration
    % Consider a minimization problem
    opt.min_objective = @(y_edge_list) FoM_and_grad(y_edge_list, h, syst, sim_info, channels, other_derivatives, options);

    % Run the optimization
    % y_edge_opt:        Optimized optimization variables
    % FoM_min:            Minimized FoM
    % retcode:              Returned value from NLopt, showing why the optimization stops
    [y_edge_opt, FoM_min, retcode] = nlopt_optimize(opt, y_edge_list_init);
else
    using_fmincon = false;
        
    if ~using_fmincon   % Use the gradient-based method with backtracking line search (BLS) to do the optimization
        % Parameters used in backtracking line search
        gamma_max = 2e-4;        % Maximum learning rate [micron^2]
        tau = 1/2;                         % Shrink factor of the learning rate (<1)
        c1 = 1e-2;                        % c1 in the Armijo condition
        % Save BLS parameters (will use in the computation of FoM and gradient later)
        BLS_parameter.min_feature = min_feature;
        BLS_parameter.gamma_max = gamma_max;
        BLS_parameter.tau = tau;
        BLS_parameter.c1 = c1;
        BLS_parameter.constraint.y_low = y_low;
        BLS_parameter.constraint.y_high = y_high;
        BLS_parameter.constraint.num_eval_max = num_eval_max;
        BLS_parameter.constraint.ftol_abs = ftol_abs;
        
        % Run the optimization
        % Output the optimized edge positions, and the FoM from each iteration
        [y_edge_opt, FoM_list, kkk] = gradient_descent_BLS(y_edge_list_init, h, syst, sim_info, channels, other_derivatives, BLS_parameter, options);
        FoM_min = FoM_list(kkk);   % Minimized FoM

    else  % Use the fmincon function in Matlab to do the optimization
        FoM_and_grad_fmincon = @(y_edge_list) FoM_and_grad(y_edge_list, h, syst, sim_info, channels, other_derivatives, options);

        % Fabrication constraints: y_{k+1} - y_k = dy >= min_feature
        A_con = diag(ones(num_edge_change,1), 0) + diag(-1*ones(num_edge_change-1,1), 1);
        b_con = [-min_feature*ones(num_edge_change-1,1); 0];

        % Set up fmincon
        options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'Display','iter','MaxIterations',num_eval_max,'OptimalityTolerance',ftol_abs);

        % The lower and upper bounds of each optimization variable
        lower_bounds = (y_low:min_feature:(y_low+min_feature*(num_edge_change-1))).';
        upper_bounds = ((y_high-min_feature*(num_edge_change-1)):min_feature:y_high).';

        % Run the optimization
        % Output the optimized edge positions, and the minimized FoM
        [y_edge_opt,FoM_min,output] = fmincon(FoM_and_grad_fmincon, y_edge_list_init, A_con, b_con, [], [], lower_bounds, upper_bounds, [], options);
    end    
end

t2_tot = clock;
t_tot = etime(t2_tot, t1_tot);
