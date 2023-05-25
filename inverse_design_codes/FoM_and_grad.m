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
%          A structure that specifies the system, used to build the FDFD matrix A in MESTI.
%          See the helping documents of MESTI for more details.
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
%          left/right of the system. See the helping documents of MESTI for more details.  
%          It contains the following fields:
%               channels.L.B_L (2D matrix):                              Source profiles on the left
%               channels.L.N_L (numeric scalar):                       Number of channels on the left
%               channels.L.pos (four-element integer vector):  Position of block source on the left
%               channels.R.B_R (2D matrix):                             Source profiles on the right
%               channels.R.N_R (numeric scalar):                     Number of channels on the right
%               channels.R.pos (four-element integer vector): Position of block source on the right
%               channels.sqrt_nu_L (1-by-N_L row vector):     Flux-normalization prefactor sqrt(nu) on the left
%               channels.sqrt_nu_R (1-by-N_R row vector):    Flux-normalization prefactor sqrt(nu) on the right
%           The two following entries are only used when the input/output is compressed (options.compress
%           = true)
%               channels.L.M_L (numeric scalar):                    Number of channels on the left before padding
%               channels.L.M_R (numeric scalar):                    Number of channels on the right before padding
%   other_derivatives (structure array; required)
%          A structure that contains the function handles of the figure of merit (FoM), \partial f/
%          \partial pk, and \partial f/\partial T. It contains the following fields:
%          other_derivatives.FOM:     Function handle of the FoM
%          other_derivatives.pfppk:   Function handle of \partial f/\partial pk
%          other_derivatives.dfdT:     Function handle of \partial f/\partial T
%   options (scalar structure; optional):                               
%          A structure that specifies the options of computation.
%          It can contain the following fields:
%          options.symmetry (logical scalar, defaults to false):
%                Whether or not a metasurface is symmetric with respect to its central plane.
%          options.method (character vector, defaults to APF): 
%                The solution method. Available choices are (case-insensitive):
%                'APF':       Augmented partial factorization
%                'adjoint':  Conventional adjoint method
%          options.N_sub (numeric scalar, defaults to 1):
%                Divide one large APF computation yielding both the transmission matrix and 
%                the gradient into N_sub sub-APF computations to save time and memory.
%                N_sub must be an integer scalar > 0.
%          options.compress (logical scalar, defaults to false):
%                Whether to compress the inputs and outputs to reduce the number of
%                nonzero elements in B and C such that nnz(B) < nnz(A), and thus reduce 
%                the computing time and memory usage of APF.
%          options.compress_par (scalar structure, only used when options.compress = true):
%                Parameters for input/output matrix compression if enabled.
%                   use_Hann_window (defaults to true):  Whether or not to use the Hann window function for the scaling
%                       to reduce the compression error.
%                   pad_ratio_L (defaults to 0.5): (number of channels to compute)/(number of extra channels to pad)
%                       at the left side of the system. 
%                   pad_ratio_R (equals to pad_ratio_L by default): (number of channels to compute)/(number of extra channels to pad)
%                       at the right side of the system.
%                   trunc_width_over_lambda_L (default to 10): (truncation window width)/wavelength
%                       at the left side of the system.
%                   trunc_width_over_lambda_R (equals to trunc_width_over_lambda_L by default): (truncation window width)/wavelength
%                       at the right side of the system.

%   === Output Arguments ===
%   FoM (numeric scalar):                              
%          Figure of merit or objective function
%   dfdpk (row vector):
%          Gradient of the figure of merit with respect to optimization variables

if nargin < 6
    error('Not enough input arguments.');
end

if ~isrow(y_edge_list)
    error('y_edge_list must be a row vector.'); 
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

if ~isfield(options, 'compress') || isempty(options.compress)
    options.compress = false;
elseif ~(islogical(options.compress) && isscalar(options.compress))
    error('options.compress must be a logical scalar, if given.');
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
if options.compress && strcmpi(options.method, 'adjoint')
    warning('Input/output matrix division is not used when options.method = adjoint.');
end

if strcmpi(options.method, 'adjoint') && nargout == 1
    warning('If gradient is not needed, use APF method to efficiently simulate multi-input problems.')
end

if options.compress
    if ~isfield(options, 'compress_par') || isempty(options.compress_par)
        options.compress_par = struct;
    end
    if ~isfield(options.compress_par, 'use_Hann_window') || isempty(options.compress_par.use_Hann_window)
         options.compress_par.use_Hann_window = true;
         warning('options.compress_par.use_Hann_window = true by default, if not specified.')
    end
    if ~isfield(options.compress_par, 'pad_ratio_L') || isempty(options.compress_par.pad_ratio_L)
         options.compress_par.pad_ratio_L = 0.5;
         warning('options.compress_par.pad_ratio_L = 0.5 by default, if not specified.')
    end
    if ~isfield(options.compress_par, 'pad_ratio_R') || isempty(options.compress_par.pad_ratio_R)
         options.compress_par.pad_ratio_R = options.compress_par.pad_ratio_L;
         warning('options.compress_par.pad_ratio_R = options.compress_par.pad_ratio_L by default, if not specified.')
    end
    if ~isfield(options.compress_par, 'trunc_width_over_lambda_L') || isempty(options.compress_par.trunc_width_over_lambda_L)
         options.compress_par.trunc_width_over_lambda_L = 10;
         warning('options.compress_par.trunc_width_over_lambda_L = 10 by default, if not specified.')
    end
    if ~isfield(options.compress_par, 'trunc_width_over_lambda_R') || isempty(options.compress_par.trunc_width_over_lambda_R)
         options.compress_par.trunc_width_over_lambda_R = options.compress_par.trunc_width_over_lambda_L;
         warning('options.compress_par.trunc_width_over_lambda_R = options.compress_par.trunc_width_over_lambda_L by default, if not specified.')
    end
else
    if isfield(options, 'compress_par')
        warning('Ignore options.compress_par if options.compress = false.')
    end
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
nz_tot = nz + nz_extra_left + nz_extra_right; 
clear sim_info

N_L = channels.L.N_L;
N_R = channels.R.N_R;
M = N_L + N_R;  % Total number of channels from both two sides of the metasurface

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
    num_pixel_change = num_edge_change*2*nz;          
else   % General metasurface
    % Get the y coordinates of all ridge edges of the whole metasurface
    y_edge_whole_list = y_edge_list;
    
    % Number of pixels changed by all optimization variables
    num_pixel_change = num_edge_change*nz; 
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
    % Get dAdpk and do a symmetric singular value decomposition (SVD)
    % dAdpk = U_k*Sigma_k*U_k.'
    % Each column of U_k (ie, u_i) has only 1 nonzero element = 1
    % diag(Sigma_k) = dAdpk at pixels specified by u_i
    Sigma = zeros(num_pixel_change, 1);
    ind_interface_list = zeros(num_pixel_change, 2);
    n_sigma = 0;

    % If the metasurface is symmetric with respect to its central plane
    if options.symmetry     
        % Get the non-zero elements of dAdpk at the ridge edges that are altered by p_k
        % obtained from the expression of subpixel smoothing
        % Symmetric SVD
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
    
                % Indices list of interface and corner pixels for all p_k (used to build matrix U)
                % 1st column: row indices;   2nd column: column indices
                ind_interface_list(n_sigma+(1:2*nz), :) = [row_l, col_l; row_r, col_r];
    
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
    
                % Indices list of interface and corner pixels for all p_k (used to build matrix U)
                % 1st column: row indices;   2nd column: column indices
                ind_interface_list(n_sigma+(1:nz), :) = [row, col];
    
                n_sigma = n_sigma + nz;
        end
    end

    % Use APF method to compute the FoM and its gradient
    if strcmpi(options.method, 'APF')    
        % Compute the gradient using APF method with matrix division, see the
        % inverse design paper for more details
        N_sub = options.N_sub;
        n_mod = mod(num_edge_change, N_sub);
        % Number of columns for each subset of matrix U, ie, U_(nnn)
        if floor(num_edge_change/N_sub) > 1
            num_design_var = floor(num_edge_change/N_sub)*ones(N_sub, 1) + [ones(n_mod, 1); zeros(N_sub-n_mod, 1)];
        else
            N_sub = num_edge_change;
            num_design_var = ones(N_sub,1);
            warning('Reduce N_sub from %d to %d such that each sub-APF computation yields the gradient for one design variable.', options.N_sub, N_sub);
        end

        % Build input-profile B_tilde = [B, U_(nnn)]
        B_struct_tilde = struct('pos', cell(1,3), 'data', cell(1,3));
    
        % Input plane waves
        % Note that even though we only need the transmission matrix with input
        % from the left (B_L) and output from the right (B_R), here we also include input channels from the right (same
        % as the output channels of interest). This allows us to make matrix K =
        % [A,B;C,0] symmetric. The computing time and memory usage of such partial
        % factorization can be reduced when matrix K is symmetric.
        B_struct_tilde(1).pos = channels.L.pos;
        B_struct_tilde(2).pos = channels.R.pos;
        B_struct_tilde(1).data = B_L;
        B_struct_tilde(2).data = B_R;
        clear B_L B_R
        
        pfppk = other_derivatives.pfppk(y_edge_list);  % Get pfppk
        % Get the gradient of FoM with respect to optimization variables
        dfdpk = zeros(1,num_edge_change);
        for nnn = 1:N_sub
            if options.symmetry
                num_col_U_sub = num_design_var(nnn)*nz*2;
            else
                num_col_U_sub = num_design_var(nnn)*nz;
            end
    
            % Build the block of sparse matrix U_(nnn)
            if nnn > 1
                if options.symmetry
                    ind_sub_interface = sum(num_design_var(1:nnn-1)*nz*2) + (1:num_col_U_sub);
                else
                    ind_sub_interface = sum(num_design_var(1:nnn-1)*nz) + (1:num_col_U_sub);
                end
            else
                 ind_sub_interface = 1:num_col_U_sub;
            end
    
            % Add U_(nnn) [additional inputs being the singular vectors from the design parameters] to B_tilde
            % Note that the block of U_(nnn) starts from the x pixel at the front surface of metasurface,
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
            if nnn == 1
                T_FOV = S_all((N_L+1):M, 1:N_L);
            end
            CAU_sub = S_all((N_L+1):M, (M+1):end); % Only need C with N_R channels on the right of the metasurface
            VAB_sub = S_all((M+1):end, 1:N_L);         % Only need B with N_L channels on the left of the metasurface
            clear S_all

            % decompress
            if options.compress
            % Here we do:
            % (1) centered  fft along dimension 1, equivalent to multiplying F_R on the left, and
            % (2) centered ifft along dimension 2, equivalent to multiplying inv(F_L) on the right.
                 if nnn == 1
                     T_FOV = fftshift((ifft(ifftshift(fftshift((fft(ifftshift(T_FOV,1),[],1)),1),2),[],2)),2)/sqrt(N_R/N_L);
                 end
                 CAU_sub = fftshift((fft(ifftshift(CAU_sub,1),[],1)),1)/sqrt(N_R);
                 VAB_sub = fftshift((ifft(ifftshift(VAB_sub,2),[],2)),2)*sqrt(N_L);

                 % Remove the extra channels we padded earlier.
                 if nnn == 1
                     M_L_pad_half = round((N_L - channels.L.M_L)/2);
                     M_R_pad_half = round((N_R - channels.R.M_R)/2);
                     ind_L = M_L_pad_half + (1:channels.L.M_L);
                     ind_R = M_R_pad_half + (1:channels.R.M_R);
                     T_FOV = T_FOV(ind_R,ind_L);
                 end
                 CAU_sub = CAU_sub(ind_R,:);
                 VAB_sub = VAB_sub(:,ind_L);

                 % Undo the diagonal scaling, per Eq (S37) of the APF paper
                 if options.compress_par.use_Hann_window
                    if nnn == 1
                        a_L = (-round((channels.L.M_L-1)/2):round((channels.L.M_L-1)/2));   % row vector
                        a_R = (-round((channels.R.M_R-1)/2):round((channels.R.M_R-1)/2)).'; % column vector
                        q_inv_L = 2./(1+cos((2*pi/N_L)*a_L)); % row vector
                        q_inv_R = 2./(1+cos((2*pi/N_R)*a_R)); % column vector
                        T_FOV = q_inv_R.*T_FOV.*q_inv_L; % use implicit expansion
                     end
                        CAU_sub = q_inv_R.*CAU_sub;
                        VAB_sub = VAB_sub.*q_inv_L;
                 end
            else
                    % Extract C*inv(A)*B, C*inv(A)*U, and U.'*inv(A)*B
                    % Generally, the transverse mode profiles of inputs/outputs are complex, and C = B'.
                    % Note that transverse mode profiles after compression are real-valued, thus C = transpose(B).
                    % Here we impose C = transpose(B) to make
                    % the augmented matrix K symmetric such that APF becomes more efficienct.
                    % Since taking the complex conjugate of the profile is equivalent to flipping the sign of ky or
                    % flipping the sign of the channel index, we need to reverse the ordering of the output channels in C
                    % after APF computation. More details can be found in Supplementary Sec.3 of the APF paper.
                    CAU_sub = CAU_sub(N_R:-1:1, :);   % Reorder C*inv(A)*U [because we use C = transpose(B)]
                    T_FOV = flipud(T_FOV);
            end
            
            % Multiply flux-normalied prefactors to C*inv(A)*U and U.'*inv(A)*B
            CAU_sub = reshape(sqrt_nu_R, [], 1).*CAU_sub;
            VAB_sub = VAB_sub.*sqrt_nu_L*(-2i);                 
            if nnn ==1
                T_FOV = (-2i)*reshape(sqrt_nu_R, [], 1).*T_FOV.*sqrt_nu_L;       % Multiply corresponding prefactors
                dfdT = other_derivatives.dfdT(T_FOV);             % Get dfdT
            end
            
           if nnn > 1
               kk_init = sum(num_design_var(1:nnn-1));
           else
               kk_init = 0;
           end
           if options.symmetry    % Symmetric metasurface              
                for kk = 1:num_design_var(nnn)
                   dTdpk = -CAU_sub(:,(kk-1)*2*nz+(1:2*nz))*diag(Sigma((kk_init+kk-1)*2*nz+(1:2*nz)))*VAB_sub((kk-1)*2*nz+(1:2*nz),:); % Eq.(3) of the inverse design paper
                   dfdpk(kk_init + kk) = pfppk + sum(reshape(2*real(dfdT.*dTdpk), [], 1)); % Eq.(1) of the inverse design paper
                end
            else   % General metasurface    
                for kk = 1:num_design_var(nnn)
                  dTdpk = -CAU_sub(:,(kk-1)*nz+(1:nz))*diag(Sigma((kk_init+kk-1)*nz+(1:nz)))*VAB_sub((kk-1)*nz+(1:nz),:); % Eq.(3) of the inverse design paper
                  dfdpk(kk_init + kk) = pfppk + sum(reshape(2*real(dfdT.*dTdpk), [], 1)); % Eq.(1) of the inverse design paper
                end
           end
            clear CAU_sub VAB_sub dTdpk
        end
    
        clear Sigma

    elseif strcmpi(options.method, 'adjoint')
        % First, we build matrix C
        m2 = channels.R.pos(1) + channels.R.pos(3) - 1;  % last index in y
        n2 = channels.R.pos(2);                                         % last index in z
        % Stack reshaped B_R with zeros to build matrix
        nyz_before = sub2ind([ny_tot, nz_tot], channels.R.pos(1), channels.R.pos(2)) - 1;
        nyz_after  = ny_tot*nz_tot - sub2ind([ny_tot, nz_tot], m2, n2);
        % Generally, C = B'
        C_matrix = reshape(sqrt_nu_R, [], 1).*[sparse(N_R, nyz_before), B_R', sparse(N_R, nyz_after)];
        
        sum_all_pixel = sparse(num_pixel_change,1);
        ind_pixel_sum = sub2ind([ny_tot,nz_tot], ny_extra_low + ind_interface_list(:,1), nz_extra_left + ind_interface_list(:,2));
        T_FOV = zeros(N_R, N_L);
        for ii = 1:N_L % Loop over all N_L inputs
            % Build input-profile B with channels from left of the metasurface
            B_struct = struct('pos', {}, 'data', {});
            B_struct(1).pos = channels.L.pos;
            B_struct(1).data = B_L(:,ii);
            
            C = [];         % Compute the field information inv(A)*B
            D = [];         % We only need the transmission matrix, for which D=0
            
            opts_adjoint.method = 'FS';   % Factorize A=L*U, solve for X=inv(A)*B with forward and backward substitutions
            opts_adjoint.clear_BC = true;
            opts_adjoint.prefactor = (-2i)*sqrt_nu_L(ii); % Multiply factors related to matrix B [see Supplementary Eqs.27,29 of the APF paper]
    
            % Solve the forward problem for inv(A)*B
            % Prefactors will be multiplied later.
            AB_forward = mesti(syst, B_struct, C, D, opts_adjoint);
            
            % Multiply matrix C to inv(A)*B, and get the transmission matrix
            T_FOV(:,ii) = C_matrix*AB_forward(:);
            
            % Build the adjoint source C_adj [see Supplementary Sec.1 of the inverse design paper]
            dfdT = other_derivatives.dfdT(T_FOV);  % Get dfdT
            C_adj = C_matrix.'*dfdT(:,ii);
            
            opts_adjoint.prefactor = 1;
            % Solve the adjoint problem inv(A)*C_adj.'
            AB_adjoint = mesti(syst, C_adj, C, D, opts_adjoint);
            
            sum_all_pixel = sum_all_pixel + 2*real(reshape(AB_adjoint(ind_pixel_sum),[],1).*reshape(AB_forward(ind_pixel_sum),[],1));
            clear AB_forward AB_adjoint
        end
        clear C_matrix
        pfppk = other_derivatives.pfppk(y_edge_list);  % Get pfppk
    
        % Get the gradient of FoM with respect to optimization variables
        dfdpk = zeros(1,num_edge_change);
        if options.symmetry    % Symmetric metasurface              
            for kk = 1:num_edge_change
               dfdpk(kk) = pfppk - sum(Sigma((kk-1)*2*nz+(1:2*nz)).*sum_all_pixel((kk-1)*2*nz+(1:2*nz))); % Eq.(1) of the inverse design paper
            end
        else   % General metasurface    
            for kk = 1:num_edge_change
              dfdpk(kk) = pfppk - sum(Sigma((kk-1)*nz+(1:nz)).*sum_all_pixel((kk-1)*nz+(1:nz))); % Eq.(1) of the inverse design paper
            end
        end
        clear sum_all_pixel Sigma
    end
    
    % Compute the FoM
    FoM = other_derivatives.FOM(T_FOV);
else  % Only compute the FoM using APF
    opts.prefactor = -2i;        % Multiply (-2i) to C*inv(A)*B
    
    % Build input-profile B
    B_struct = struct('pos', cell(1,2), 'data', cell(1,2));

    % Input plane waves
    % Note that even though we only need the transmission matrix with input
    % from the left (B_L) and output from the right (B_R), here we also include input channels from the right (same
    % as the output channels of interest). This allows us to make matrix K =
    % [A,B;C,0] symmetric. The computing time and memory usage of such partial
    % factorization can be reduced when matrix K is symmetric.
    B_struct(1).pos = channels.L.pos;
    B_struct(2).pos = channels.R.pos;
    B_struct(1).data = B_L;
    B_struct(2).data = B_R;

    C = 'transpose(B)';         % Specify C = transpose(B)
    D = [];                           % We only need the transmission matrix, for which D=0
    
    % Calculate C*inv(A)*B only
    % Prefactors will be multiplied later.
    S = mesti(syst, B_struct, C, D, opts);
    T_FOV = S((N_L+1):M, 1:N_L);      % Only need C with N_R channels on the right of the metasurface
                                                        % Only need B with N_L channels on the left of the metasurface
    clear S
    
    % decompress
    if options.compress
        % Here we do:
        % (1) centered  fft along dimension 1, equivalent to multiplying F_R on the left, and
        % (2) centered ifft along dimension 2, equivalent to multiplying inv(F_L) on the right.
        T_FOV = fftshift((ifft(ifftshift(fftshift((fft(ifftshift(T_FOV,1),[],1)),1),2),[],2)),2)/sqrt(N_R/N_L);
        
        % Remove the extra channels we padded earlier.
        M_L_pad_half = round((N_L - channels.L.M_L)/2);
        M_R_pad_half = round((N_R - channels.R.M_R)/2);
        ind_L = M_L_pad_half + (1:channels.L.M_L);
        ind_R = M_R_pad_half + (1:channels.R.M_R);
        T_FOV = T_FOV(ind_R,ind_L);
        
        % Undo the diagonal scaling, per Eq (S37) of the APF paper
        if options.compress_par.use_Hann_window
            a_L = (-round((channels.L.M_L-1)/2):round((channels.L.M_L-1)/2));   % row vector
            a_R = (-round((channels.R.M_R-1)/2):round((channels.R.M_R-1)/2)).'; % column vector
            q_inv_L = 2./(1+cos((2*pi/N_L)*a_L)); % row vector
            q_inv_R = 2./(1+cos((2*pi/N_R)*a_R)); % column vector
            T_FOV = q_inv_R.*T_FOV.*q_inv_L; % use implicit expansion
        end
    else
        % Generally, the transverse mode profiles of inputs/outputs are complex, and C = B'.
        % Note that transverse mode profiles after compression are real-valued, thus C = transpose(B).
        % Here we impose C = transpose(B) to make
        % the augmented matrix K symmetric such that APF becomes more efficienct.
        % Since taking the complex conjugate of the profile is equivalent to flipping the sign of ky or
        % flipping the sign of the channel index, we need to reverse the ordering of the output channels in C
        % after APF computation. More details can be found in Supplementary Sec.3 of the APF paper.
        T_FOV = flipud(T_FOV);
    end
    T_FOV = reshape(sqrt_nu_R, [], 1).*T_FOV.*sqrt_nu_L;  % Multiply flux-normalization factors

    % Compute the FoM
    FoM = other_derivatives.FOM(T_FOV);
end

end