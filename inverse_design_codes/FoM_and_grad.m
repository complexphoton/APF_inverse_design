function [FoM, dfdpk] = FoM_and_grad(y_edge_list, h, syst, sim_info, channels, other_derivatives, options)

% FOM_AND_GRAD computes the FoM and its gradient with respect to optimization
% variables for a 2D metasurface under TM polarization, ie, Ex(y,z).
% It works when the FoM is a real scalar with contributions from all channels listed in the structure channels,
% and the optimization variable is a real-valued vector.

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
%               sim_info.num_pixel.nz:                    Pixels of metasurfaces in the z direction
%               sim_info.num_pixel.ny:                    Pixels of metasurfaces in the y direction
%               sim_info.num_pixel.nz_extra_left:    Pixels added on the left side in the z direction
%               sim_info.num_pixel.nz_extra_right:  Pixels added on the right side in the z direction
%               sim_info.num_pixel.ny_extra_high:   Pixels added on the upper side in the y direction
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
%   options (scalar structure; optional, defaults to an empty struct):                               
%          A structure that specifies the options of computation; defaults to an
%          empty structure. It can contain the following fields:
%          options.symmetry (logical scalar, defaults to false):
%                Whether or not a metasurface is symmetric with respect to its central plane.
%          options.method (character vector, defalts to APF): 
%                The solution method. Available choices are (case-insensitive):
%                'APF':       Augmented partial factorization
%                'adjoint':  Adjoint method
%          options.N_sub (numeric scalar, defaults to 1):
%                Divide one large APF computation yielding both the transmission matrix and 
%                the gradient into N_sub small APF computations to save time and memory.
%                N_sub must be an integer scalar > 0.

%   === Output Arguments ===
%   FoM (numeric scalar):                              
%          Figure of merit
%   dfdpk (row vector):
%          Gradient of the figure of merit with respect to optimization variables

if nargin < 6
    error('Not enough input arguments.');
end

if ~isrow(y_edge_list)
    error('y_edge_list must be row vectors.'); 
end

if ~(isreal(channels.L.pos) && isvector(channels.L.pos) && numel(channels.L.pos)==4 && isequal(channels.L.pos, round(channels.L.pos)) && min(channels.L.pos)>0)
    error('channels.L.pos must be a four-element integer vector.')
end

if ~(isreal(channels.R.pos) && isvector(channels.R.pos) && numel(channels.R.pos)==4 && isequal(channels.R.pos, round(channels.R.pos)) && min(channels.R.pos)>0)
    error('channels.R.pos must be a four-element integer vector.')
end

if ~(numel(fieldnames(other_derivatives))==3 && isa(other_derivatives.FOM, 'function_handle') && isa(other_derivatives.dfdT, 'function_handle') && isa(other_derivatives.pfppk, 'function_handle'))
    error('other_derivatives must be a three-element array containing function handles.')
end

if ~isfield(options, 'symmetry') || isempty(options.symmetry)
    options.symmetry = false;
elseif ~(islogical(options.symmetry) && isscalar(options.symmetry))
    error('options.symmetry must be a logical scalar, if given.');
end

if ~isfield(options, 'method') || isempty(options.method)
    options.method = 'APF';
elseif ~(strcmpi(options.method, 'APF') || strcmpi(options.method, 'adjoint'))
    error('options.method must be either APF or adjoint (case-insensitive).');
end

if strcmpi(options.method, 'APF') && nargout > 1
    if (~isfield(options, 'N_sub') || isempty(options.N_sub))
        options.N_sub = 1;
    elseif ~(isreal(options.N_sub) && isscalar(options.N_sub) && isequal(options.N_sub, round(options.N_sub)) && options.N_sub>0)
        error('options.N_sub must be an integer real scalar > 0, if given.');
    end
end

if strcmpi(options.method, 'adjoint') && isfield(options, 'N_sub')
    warning('N_sub is not used when options.method = adjoint.');
end

if strcmpi(options.method, 'adjoint') && nargout == 1
    warning('If gradient is not needed, use APF method for efficient simulation of multi-input problems.')
end

% Extract refractive indices 
n_bg = sim_info.RI.n_bg;
n_sub = sim_info.RI.n_sub;
n_ridge = sim_info.RI.n_ridge;
epsilon_L = n_sub^2;        % Relative permittivity on the left
epsilon_R = n_bg^2;         % Relative permittivity on the right and on the top & bottom

ny = sim_info.num_pixel.ny;
nz = sim_info.num_pixel.nz;
nz_extra_left = sim_info.num_pixel.nz_extra_left;
nz_extra_right = sim_info.num_pixel.nz_extra_right;
ny_extra_low = sim_info.num_pixel.ny_extra_low;
ny_extra_high = sim_info.num_pixel.ny_extra_high;
% Number of pixels for the whole simulation domain in the y direction
ny_tot = ny + ny_extra_low + ny_extra_high;        
% Number of pixels for the whole simulation domain in the z direction
% nz_tot = nz + nz_extra_left + nz_extra_right;  

N_L = channels.L.N_L;
N_R = channels.R.N_R;
M = N_L + N_R;  % Total number of input (output) channels coming from both two sides of the metasurface

B_L = channels.L.B_L;
B_R = channels.R.B_R;
% The flux-normalization prefactor sqrt(nu)
sqrt_nu_L = channels.sqrt_nu_L;
sqrt_nu_R = channels.sqrt_nu_R;

% Number of edges (ie, optimization variables)
num_edge_change = length(y_edge_list);   

if options.symmetry  % Symmetric metasurface    
    % Get the y coordinates of all ridge edges of the whole metasurface based on symmetry
    y_edge_whole_list = [y_edge_list, fliplr(-y_edge_list)];
    
    % Number of pixels changed by all optimization variables
    num_pixel_change = num_edge_change*2*sim_info.num_pixel.nz;          
else   % General metasurface
    % Get the y coordinates of all ridge edges of the whole metasurface
    y_edge_whole_list = y_edge_list;
    
    % Number of pixels changed by all optimization variables
    num_pixel_change = num_edge_change*sim_info.num_pixel.nz; 
end

wavelength = syst.wavelength;
dx = syst.dx;
k0dx = 2*pi/wavelength*dx; % Dimensionless frequency k0*dx

% Input edge coordinates, output the permittivity profile of the metasurface and
% y indices of the edge pixels
% edge_pixel_ind(ii,1): y index of left edge of ii-th ridge; edge_pixel_ind(ii,2): y index of right edge of ii-th ridge
[epsilon, edge_pixel_ind] = build_epsilon_pos(dx, n_bg, n_ridge, y_edge_whole_list, h, ny);

% Include homogeneous space and PML to the permittivity profile.
syst.epsilon = [epsilon_L*ones(ny_tot, nz_extra_left), ...
   [epsilon_R*ones(ny_extra_low,nz); epsilon; epsilon_R*ones(ny_extra_high,nz)], ...
    epsilon_R*ones(ny_tot, nz_extra_right)];

% Compute the FoM and its gradient dfdpk
if nargout > 1
    % Get dAdpk and do SVD analytically
    % dAdpk = U_k*Sigma_k*U_k.'
    % Each column of U_k (ie, u_i) has only 1 nonzero element = 1
    % diag(Sigma_k) = dAdpk at pixels specified by u_i
    Sigma = zeros(num_pixel_change, 1);
    ind_interface_list = zeros(num_pixel_change, 2);
    n_sigma = 0;

    % If the metasurface is symmetric with respect to its central plane
    if options.symmetry     
        % Get the non-zero elements of dAdpk at the ridge edges that are altered by p_k
        % Do SVD analytically, obtained from the expression of subpixel smoothing
        for ii = 1:num_edge_change
             ind1 = [ceil(ii/2), num_edge_change-ceil(ii/2)+1];        % Pick the two ridges that are changed by p_k
                                                                                                  % One on the left-half side of the metasurface, the other on the right-half side.
    
             if mod(ii,2) == 0 % right edge on the left-half side; left edge on the right-half side 
                % Row and column indices of y_k (left-half side)
                [row_l, col_l] = meshgrid(edge_pixel_ind(ind1(1), 2), (1:nz));
    
                % Row and column indices of y_{2N-k+1} (right-half side)
                [row_r, col_r] = meshgrid(edge_pixel_ind(ind1(2), 1), (1:nz));
                    
                % dAdpk = dAdy_k + dAdy_{2N-k+1}*(dy_{2N-k+1})/dy_k
                % = dAdy_k - dAdy_{2N-k+1}
                % The right edge (dpk > 0: the ridge gets wider) on the left-half side
                % Because an additional (-1) prefactor is multiplied for y_{2N-k+1} on the right-half side, 
                % the sign difference between left and right edges is cancelled.
                % Interface pixel
                Sigma(n_sigma+[1:(nz-1), (nz+1):(2*nz-1)]) = -k0dx^2*(n_ridge^2 - n_bg^2)/dx;            
                % Corner pixel
                Sigma(n_sigma+[nz,2*nz]) = -k0dx^2*((n_ridge^2 - n_bg^2)*(1-(nz-h/dx)))/dx;      
             else % left edge for the left-half side; right edge for the right-half side
                % Row and column indices of y_k (left-half side)
                [row_l, col_l] = meshgrid(edge_pixel_ind(ind1(1), 1), (1:nz));
    
                % Row and column indices of y_{2N-k+1} (right-half side)
                [row_r, col_r] = meshgrid(edge_pixel_ind(ind1(2), 2), (1:nz));
    
                % dAdpk = dAdy_k + dAdy_{2N-k+1}*(dy_{2N-k+1})/dy_k
                % = dAdy_k - dAdy_{2N-k+1}
                % The left edge (dpk > 0: the ridge gets thinner) on the left-half side
                % Because an additional (-1) prefactor is multiplied for y_{2N-k+1} on the right-half side, 
                % the sign difference between left and right edges is cancelled.
                % Interface pixel
                Sigma(n_sigma+[1:(nz-1), (nz+1):(2*nz-1)]) = k0dx^2*(n_ridge^2 - n_bg^2)/dx;            
                % Corner pixel
                Sigma(n_sigma+[nz,2*nz]) = k0dx^2*((n_ridge^2 - n_bg^2)*(1-(nz-h/dx)))/dx;    
             end
                % Indices of interface and corner pixels for one p_k
                % 1st column: row indices;   2nd column: column indices
                ind_1l = [row_l(1:end-1), col_l(1:end-1)];   % interface of y_k
                ind_1c = [row_l(end), col_l(end)];                % corner of y_k
                ind_2l = [row_r(1:end-1), col_r(1:end-1)];  % interface of y_{2N-k+1}
                ind_2c = [row_r(end), col_r(end)];               % corner of y_{2N-k+1}
    
                % Indices list of interface and corner pixels for all p_k (used when we build U)
                % 1st column: row indices;   2nd column: column indices
                ind_interface_list(n_sigma+(1:2*nz), :) = [ind_1l; ind_1c; ind_2l; ind_2c];
    
                n_sigma = n_sigma + 2*nz;
        end
    % General metasurface without mirror symmetry with respect to its central plane
    else     
        for ii = 1:num_edge_change
             ind1 = ceil(ii/2);        % Pick the ridge changed by p_k
    
             if mod(ii,2) == 0 % right edge
                % Row and column indices of y_k
                [row, col] = meshgrid(edge_pixel_ind(ind1, 2), (1:nz));
    
                % The right edge (dpk > 0: the ridge gets wider) 
                % Interface pixel
                Sigma(n_sigma+(1:(nz-1))) = -k0dx^2*(n_ridge^2 - n_bg^2)/dx;            
                % Corner pixel
                Sigma(n_sigma+nz) = -k0dx^2*((n_ridge^2 - n_bg^2)*(1-(nz-h/dx)))/dx;    
             else % left edge
                % Row and column indices of y_k
                [row, col] = meshgrid(edge_pixel_ind(ind1, 1), (1:nz));
    
                % The left edge (dpk > 0: the ridge gets thinner)
                % Interface pixel
                Sigma(n_sigma+(1:(nz-1))) = k0dx^2*(n_ridge^2 - n_bg^2)/dx;            
                % Corner pixel
                Sigma(n_sigma+nz) = k0dx^2*((n_ridge^2 - n_bg^2)*(1-(nz-h/dx)))/dx;    
             end
                % Indices of interface and corner pixels for one p_k
                % 1st column: row indices;   2nd column: column indices
                ind_l = [row(1:end-1), col(1:end-1)];   % interface of y_k
                ind_c = [row(end), col(end)];                % corner of y_k
    
                % Indices list of interface and corner pixels for all p_k (used when we build U)
                % 1st column: row indices;   2nd column: column indices
                ind_interface_list(n_sigma+(1:nz), :) = [ind_l; ind_c];
    
                n_sigma = n_sigma + nz;
        end
    end

    % Use APF method to compute the FoM and its gradient
    if strcmpi(options.method, 'APF')    
        % Compute the gradient using APF method with matrix division, see Sec.2 of the
        % inverse design paper for more details
        N_sub = options.N_sub;
        n_mod = mod(num_pixel_change, N_sub);
        % Number of columns for each subset of matrix U
        num_col_U = floor(num_pixel_change/N_sub)*ones(N_sub, 1) + [ones(n_mod, 1); zeros(N_sub-n_mod, 1)];
    
        CAU = zeros(N_R, num_pixel_change);    % C*inv(A)*U
        VAB = zeros(num_pixel_change, N_L);     % U.'*inv(A)*B
    
        for nnn = 1:N_sub  
            num_col_U_sub = num_col_U(nnn);
    
            % Build input-profile B_tilde = [B, U_(nnn)]
            B_struct_tilde = struct('pos', cell(1,3), 'data', cell(1,3));
    
            % Input plane waves
            B_struct_tilde(1).pos = channels.L.pos;
            B_struct_tilde(2).pos = channels.R.pos;
            B_struct_tilde(1).data = B_L;
            B_struct_tilde(2).data = B_R;
    
            % Build the block of sparse matrix U_(nnn)
            ind_sub_interface = zeros(num_col_U_sub,1);
            for ii = 1:num_col_U_sub
                if nnn > 1
                    ind_sub_interface(ii) = sum(num_col_U(1:nnn-1)) + ii;
                else
                    ind_sub_interface(ii) = ii;
                end
            end
    
            % Add U_(nnn) to B_tilde
            % Note that the block of U_(nnn) starts from the x pixel at the front surface of metalens,
            % which is different from the plane wave sources, located at one pixel before the front surface. 
            % The block of U_nnn has ny*nz rows and about num_pixel_change/N_sub columns
            % Each column only has one nonzero element = 1
            B_struct_tilde(3).pos = [channels.L.pos(1), channels.L.pos(2)+1, ny, nz];
            B_struct_tilde(3).data = sparse(sub2ind([ny,nz],ind_interface_list(ind_sub_interface,1),ind_interface_list(ind_sub_interface,2)), (1:num_col_U_sub).', ones(num_col_U_sub,1), ny*nz, num_col_U_sub);

            C = 'transpose(B)';         % Specify C = transpose(B)
            D = [];                           % We only need the transmission matrix, for which D=0
    
            % Calculate C*inv(A)*B, C*inv(A)*U, and U.'*inv(A)*B
            % Prefactors will be multiplied later.
            S_all = mesti(syst, B_struct_tilde, C, D);
    
            % Extract C*inv(A)*B, C*inv(A)*U, and U.'*inv(A)*B
            CAU_sub = S_all(M:-1:(N_L+1), (M+1):end);   % Extract and reorder C*inv(A)*U
            VAB_sub = S_all((M+1):end, 1:N_L);                % Extract U.'*inv(A)*B
            if nnn ==1
                T_FOV = S_all(M:-1:(N_L+1), 1:N_L);                                          % Extract and reorder C*inv(A)*B
                T_FOV = (-2i)*reshape(sqrt_nu_R, [], 1).*T_FOV.*sqrt_nu_L;       % Multiply corresponding prefactors
            end
    
            if nnn > 1
               CAU(:, sum(num_col_U(1:nnn-1))+(1:num_col_U_sub)) = reshape(sqrt_nu_R, [], 1).*CAU_sub;
               VAB(sum(num_col_U(1:nnn-1))+(1:num_col_U_sub), :) = VAB_sub.*sqrt_nu_L*(-2i);
            else
               CAU(:, 1:num_col_U_sub) = reshape(sqrt_nu_R, [], 1).*CAU_sub;
               VAB(1:num_col_U_sub, :) = VAB_sub.*sqrt_nu_L*(-2i);
            end
    
            clear CAU_sub VAB_sub S_all
        end
        
        dfdT = other_derivatives.dfdT(T_FOV);             % Get dfdT
        pfppk = other_derivatives.pfppk(y_edge_list);  % Get pfppk
        % Get the gradient of FoM with respect to optimization variables
        dfdpk = zeros(1,num_edge_change);
        if options.symmetry    % Symmetric metasurface              
            for kk = 1:num_edge_change
               dTdpk = -CAU(:,(kk-1)*2*nz+(1:2*nz))*diag(Sigma((kk-1)*2*nz+(1:2*nz)))*VAB((kk-1)*2*nz+(1:2*nz),:); % Eq.(3) of the inverse design paper
               dfdpk(kk) = pfppk + sum(reshape(2*real(dfdT.*dTdpk), [], 1)); % Eq.(1) of the inverse design paper
            end
        else   % General metasurface    
            for kk = 1:num_edge_change
              dTdpk = -CAU(:,(kk-1)*nz+(1:nz))*diag(Sigma((kk-1)*nz+(1:nz)))*VAB((kk-1)*nz+(1:nz),:); % Eq.(3) of the inverse design paper
              dfdpk(kk) = pfppk + sum(reshape(2*real(dfdT.*dTdpk), [], 1)); % Eq.(1) of the inverse design paper
            end
        end
    
        clear CAU VAB Sigma dTdpk

    elseif strcmpi(options.method, 'adjoint')
        % Build input-profile B
        B_struct_tilde = struct('pos', cell(1,2), 'data', cell(1,2));
        B_struct_tilde(1).pos = channels.L.pos;
        B_struct_tilde(2).pos = channels.R.pos;
        B_struct_tilde(1).data = B_L;
        B_struct_tilde(2).data = B_R;

        % Build projection-profile C
        C_struct_tilde = struct('pos', cell(1,2), 'data', cell(1,2));
        C_struct_tilde(1).pos = channels.R.pos;
        C_struct_tilde(1).data = B_R;
        C_struct_tilde(2).pos = [channels.L.pos(1), channels.L.pos(2)+1, ny, nz];
        C_struct_tilde(2).data = sparse(sub2ind([ny,nz],ind_interface_list(:,1),ind_interface_list(:,2)), (1:num_pixel_change).', ones(num_pixel_change,1), ny*nz, num_pixel_change);

        D = [];      % We only need the transmission matrix, for which D=0
        
        opts_adjoint.method = 'FS';
        opts_adjoint.nrhs = 1;
        opts_adjoint.clear_BC = true;

        % Calculate C*inv(A)*B, C*inv(A)*C.', U.'*inv(A)*B and U.'*inv(A)*C.'
        % Prefactors will be multiplied later.
        S_all = mesti(syst, B_struct_tilde, C_struct_tilde, D, opts_adjoint);
                    
        % Get T = C*inv(A)*B and multiply corresponding prefactors
        T_FOV = S_all(N_R:-1:1, 1:N_L);
        T_FOV = (-2i)*reshape(sqrt_nu_R, [], 1).*T_FOV.*sqrt_nu_L;

        VAB = (-2i)*S_all((N_R+1):end, 1:N_L).*sqrt_nu_L;
        VAC = S_all((N_R+1):end, M:-1:(N_L+1)).*sqrt_nu_R;

        clear S_all

        dfdT = other_derivatives.dfdT(T_FOV);             % Get dfdT
        pfppk = other_derivatives.pfppk(y_edge_list);  % Get pfppk

        % Get the gradient of FoM with respect to optimization variables
        dfdpk = zeros(1,num_edge_change);
        if options.symmetry    % Symmetric metasurface              
            for kk = 1:num_edge_change
               dTdpk = -(VAC((kk-1)*2*nz+(1:2*nz),:)).'*diag(Sigma((kk-1)*2*nz+(1:2*nz)))*VAB((kk-1)*2*nz+(1:2*nz),:); % Eq.(3) of the inverse design paper
               dfdpk(kk) = pfppk + sum(reshape(2*real(dfdT.*dTdpk), [], 1)); % Eq.(1) of the inverse design paper
            end
        else   % General metasurface    
            for kk = 1:num_edge_change
              dTdpk = -(VAC((kk-1)*nz+(1:nz),:)).'*diag(Sigma((kk-1)*nz+(1:nz)))*VAB((kk-1)*nz+(1:nz),:); % Eq.(3) of the inverse design paper
              dfdpk(kk) = pfppk + sum(reshape(2*real(dfdT.*dTdpk), [], 1)); % Eq.(1) of the inverse design paper
            end
        end
        clear VAB VAC dTdpk
    end
    
    % Compute the FoM
    FoM = other_derivatives.FOM(T_FOV);
else  % Only compute the FoM using APF
    opts.prefactor = -2i;        % Multiply (-2i) to C*inv(A)*B
    
    % Build input-profile B
    B_struct = struct('pos', cell(1,2), 'data', cell(1,2));

    % Input plane waves
    B_struct(1).pos = channels.L.pos;
    B_struct(2).pos = channels.R.pos;
    B_struct(1).data = B_L;
    B_struct(2).data = B_R;

    C = 'transpose(B)';         % Specify C = transpose(B)
    D = [];                           % We only need the transmission matrix, for which D=0
    
    % Calculate C*inv(A)*B only
    % Prefactors will be multiplied later.
    S = mesti(syst, B_struct, C, D, opts);
    
    T_FOV = S(M:-1:(N_L+1), 1:N_L);      % Extract and reorder C*inv(A)*B
    T_FOV = reshape(sqrt_nu_R, [], 1).*T_FOV.*sqrt_nu_L;  % Multiply flux-normalization factors

    clear S
    % Compute the FoM
    FoM = other_derivatives.FOM(T_FOV);
end

end