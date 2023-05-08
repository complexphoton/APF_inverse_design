function [epsilon_pos, interface_ind] = build_epsilon_pos(dx, n_bg, n_ridge, y_ridge, ridge_height, ny)

% BUILD_EPSILON_POS generates the relative permittivity profile of an array of ridges with subpixel smoothing.
%
%   === Input Arguments ===
%   dx (numeric scalar, real):
%       Grid size               
%   n_bg (numeric scalar, real or complex):
%       Refractive index of background material
%   n_ridge (numeric scalar, real or complex):
%       Refractive index of ridge material
%   y_ridge (vector, real):
%       Coordinates (y) of the two edges of all ridges
%   ridge_height (numeric scalar, real):
%       Height of the ridge
%   ny (numeric scalar, real):
%       Number of pixels along the y direction

%   === Output Arguments ===
%   epsilon_pos (numeric matrix, real or complex):
%       Discretized relative permittivity profile
%   interface_ind (two-column matrix, real)
%       Indices (y) of the structure interface 
%       interface_ind(ii,1) corresponds to the y index of left edge of the ii-th ridge
%       interface_ind(ii,2) corresponds to the y index of right edge of the ii-th ridge

nz = ceil(ridge_height/dx);                 % Number of pixels in the z direction (ie. along the height of ridges)
num_ridge = length(y_ridge)/2;          % Number of ridges

y = ((0.5:ny) - (ny/2))*dx;                    % Coordinate at the center of each pixel

% Ratio of the background & ridge along the z direction
pixel_bg_ratio_z = nz - ridge_height/dx;
pixel_ridge_ratio_z = 1 - pixel_bg_ratio_z;

% Initialize a matrix to save the y indices of the interfaces of all ridges
% The first column gives the lefe edge of the ii-th ridge
% The second column gives the right edge of the ii-th ridge
interface_ind = zeros(num_ridge, 2);

epsilon_pos = n_bg^2*ones(ny, nz);
for ii = 1:num_ridge
    
    y_edge = y_ridge((ii-1)*2+(1:2));
    
    ind_edge = interp1(y, 1:ny, y_edge, 'linear', 'extrap');
    ind_edge_round = round(ind_edge);
    if ind_edge_round(1) == ind_edge_round(2)    % Left and right edges of the ii-th ridge at the same pixel
        % Ratio of the background & ridge along the y direction
        pixel_ridge_ratio_y1 = ind_edge(2) - ind_edge(1);
        pixel_bg_ratio_y1 = 1 - pixel_ridge_ratio_y1;
        
        % Interface pixels
        epsilon_pos(ind_edge_round(1), 1:nz-1) = pixel_ridge_ratio_y1*n_ridge^2 + pixel_bg_ratio_y1*n_bg^2;
        
        % Corner pixels
        epsilon_pos(ind_edge_round(1), nz) = pixel_ridge_ratio_y1*pixel_ridge_ratio_z*n_ridge^2 + (1-pixel_ridge_ratio_y1*pixel_ridge_ratio_z)*n_bg^2;
        
        % Save the y indices of interface pixels
        interface_ind(ii,:) = ind_edge_round;

    else %% Left and right edges of the ii-th ridge at different pixels
        ind_ridge = zeros(2,1);   % Save the indices of the first and last pixels filled with ridge material for each ridge
    
        % The left edge: Ratio of the background & ridge along the y direction
        if ceil(ind_edge(1)) - ind_edge(1) < 0.5
            ind_ridge(1) = ceil(ind_edge(1))+1;
            pixel_ridge_ratio_y1 = ceil(ind_edge(1)) - ind_edge(1) + 0.5;
            pixel_bg_ratio_y1 = 1 - pixel_ridge_ratio_y1;
        else
            ind_ridge(1) = ceil(ind_edge(1));
            pixel_ridge_ratio_y1 = ceil(ind_edge(1)) - ind_edge(1) - 0.5;
            pixel_bg_ratio_y1 = 1 - pixel_ridge_ratio_y1;
        end
        
        % Take care of the case when the right edge of (ii-1)-th ridge and the left edge of ii-th ridge are at the same pixel 
        if ii > 1
            y_ridge_neighbor = y_ridge(2*ii-2);
            ind_ridge_neighbor = interp1(y, 1:ny, y_ridge_neighbor, 'linear', 'extrap');
            
            % Ratio of the background & ridge along the y direction 
            if ind_edge_round(1) == round(ind_ridge_neighbor) 
                pixel_bg_ratio_y1 = ind_edge(1) - ind_ridge_neighbor;
                pixel_ridge_ratio_y1 = 1 - pixel_bg_ratio_y1;
            end  
        end

        % The right edge: Ratio of the background & ridge along the y direction
         if ind_edge(2) - floor(ind_edge(2)) < 0.5
            ind_ridge(2) = floor(ind_edge(2)) - 1;
            pixel_ridge_ratio_y2 = ind_edge(2) - floor(ind_edge(2)) + 0.5;
            pixel_bg_ratio_y2 = 1 - pixel_ridge_ratio_y2;
        else
             ind_ridge(2) = floor(ind_edge(2));
             pixel_ridge_ratio_y2 = ind_edge(2) - floor(ind_edge(2)) - 0.5;
             pixel_bg_ratio_y2 = 1 - pixel_ridge_ratio_y2;
         end
        
        % Take care of the case when the right edge of ii-th ridge and the left edge of (ii+1)-th ridge are at the same pixel 
        if ii < num_ridge
            y_ridge_neighbor = y_ridge(2*ii+1);
            ind_ridge_neighbor = interp1(y, 1:ny, y_ridge_neighbor, 'linear', 'extrap');
            
            % Ratio of the background & ridge along the y direction 
            if ind_edge_round(2) == round(ind_ridge_neighbor) 
                pixel_bg_ratio_y2 = ind_ridge_neighbor - ind_edge(2);
                pixel_ridge_ratio_y2 = 1 - pixel_bg_ratio_y2;
            end  
        end

        if ind_ridge(1) <= ind_ridge(2)
            % Build the permittivity profile inside the ridge
            epsilon_pos(ind_ridge(1):ind_ridge(2), 1:nz-1) = n_ridge^2;
            
            % Build the permittivity profile at the top interface of each ridge (subpixel smoothing)
            epsilon_pos(ind_ridge(1):ind_ridge(2), nz) = pixel_ridge_ratio_z*n_ridge^2 + pixel_bg_ratio_z*n_bg^2;
        end
        
        % Build the permittivity profile at the left and right interfaces/corners of the ridge (subpixel smoothing)
        % Left interface
        epsilon_pos(ind_ridge(1)-1, 1:nz-1) = pixel_ridge_ratio_y1*n_ridge^2 + pixel_bg_ratio_y1*n_bg^2;
        % Left corner
        epsilon_pos(ind_ridge(1)-1, nz) = pixel_ridge_ratio_y1*pixel_ridge_ratio_z*n_ridge^2 + (1-pixel_ridge_ratio_y1*pixel_ridge_ratio_z)*n_bg^2;
        % Right interface
        epsilon_pos(ind_ridge(2)+1, 1:nz-1) = pixel_ridge_ratio_y2*n_ridge^2 + pixel_bg_ratio_y2*n_bg^2;
        % Right corner
        epsilon_pos(ind_ridge(2)+1, nz) = pixel_ridge_ratio_y2*pixel_ridge_ratio_z*n_ridge^2 + (1-pixel_ridge_ratio_y2*pixel_ridge_ratio_z)*n_bg^2;
        
        % Save the y indices of interface pixels
        interface_ind(ii,:) = [ind_ridge(1)-1, ind_ridge(2)+1];
    end
    
end


end
