function GT_visuals(A,fwMCFilePath, participantID, SSlist, Clustering, local_efficiency)
    % Working image reconstruction pipeline
    % This function loads a forward model, plots brain nodes and links,
    % and visualizes clustering coefficient and local efficiency on the brain.
    % Note that this version is only compatible with REF_GT currently
        
 
    % Parameters:
    % A - Adjacency matrix
    % fwMCFilePath - Path to the fwMC.mat file
    % participantID - ID of the participant
    % SSlist - List of short channels
    % Clustering - Matrix of clustering coefficients
    % local_efficiency - Matrix of local efficiency values
    
    
    % Load in forward model
    data = load(fwMCFilePath, 'fwmodel');
    Adot = data.fwmodel.Adot(:,:,1);  % Load the sensitivity matrix
    mesh = data.fwmodel.mesh;  % Load mesh data

    % Take coordinates based on fwMC
    CoorOpt_reg = [];
    for nchn = 1:54  % 54 because we have 54 channels
        [~, index] = max(Adot(nchn, :));
        CoorOpt_reg = [CoorOpt_reg; mesh.vertices(index, :)];
    end

%% Plot links and nodes on brain 

% This will allow us to visualize our results 
% A background image is required, use the background_graph_image.m

% This code is designed to visualize nodes and links superimposed on a
% brain image

figure()
subplot(1,3,2)
background_graph_Image;
hold on;

% now that we have the background image created we will now use it to
% populate the results

tam = 3; % Define the size of the nodes to be plotted

[x,y,z] = sphere(50,50); % Generate coordinates for a sphere, used to plot spherical nodes

% Iterate over the coordinates, placing nodes at specified locations

for N=1:size(CoorOpt_reg,1)  % Loop through each coordinate set
    
    if isempty(find(N==SSlist)==1) % Check if the current index is not a short channel
        surf(tam*x+CoorOpt_reg(N,1),tam*y+CoorOpt_reg(N,2),...
            tam*z+CoorOpt_reg(N,3),'FaceColor',...
            [0.3 0.3 0.3],'EdgeColor','none'); % Plot a gray sphere at the coordinates
    end
end

% Now we iterate over pairs of channels to draw links between them

for Nchan=1:54 % Loop through each channel
    
    for Nchan2 = 1:54 % Loop through each channel to form a pair
        
        if Nchan2>Nchan % Ensure that each pair is only considered once
            if  A(Nchan,Nchan2)>0 % Check if there's a connection between the channels
                
                P1 = CoorOpt_reg(Nchan,:); % Start point of the tube
                P2 = CoorOpt_reg(Nchan2,:); % End point of the tube
                
                DrawTubes(P1,P2,[0.3 0.3 0.3]);  % Call a function to draw a tube between the points
                
            end
            
        end
        
    end
end

% Sets the camera position for the figure, note that these specifications
% have been taken from the fNIRS course - will likely need to change it tbh
set(gca,'CameraPosition',...
    [4465.88633611231 -122.094067886442 1263.15589092293],'CameraTarget',...
    [129.6960774264 138.177489128 141.54136287528],'CameraUpVector',...
    [-0.228800956760502 0.24649272775698 0.941749147782148],'CameraViewAngle',...
    1.85583529548837,'DataAspectRatio',[1 1 1],'LineStyleOrderIndex',50,...
    'PlotBoxAspectRatio',[1 1.18303201283536 1.08555630447355]);
    
     % Define save path for the figure
    figureSavePath = fullfile('C:\Users\Hashlu\Documents\MATLAB\REF\test', ...
                              participantID, sprintf('%s_linknodes_figure.fig', participantID));
    saveas(gcf, figureSavePath);  % Save the figure

    % Plot clustering coefficient on the brain
    figure();
    subplot(1, 3, 2);
    background_graph_Image;
    hold on;

    % Normalize clustering values
    alpha = 0.9;
    aux_max = max(max(Clustering));
    x = Clustering - (alpha*aux_max);
    tam = 15 ./ (1 + exp(-4*(x)));
    Colormap = jet(1000);
    a = 999 / (max(tam) - min(tam));
    b = 1 - a * min(tam);
    Color_index = round(a * tam + b);

    % Generate spheres for each node
    [x, y, z] = sphere(50, 50);
    for N = 1:size(CoorOpt_reg, 1)
        if isempty(find(N == SSlist, 1))
            surf(tam(N)*x + CoorOpt_reg(N, 1), tam(N)*y + CoorOpt_reg(N, 2), ...
                 tam(N)*z + CoorOpt_reg(N, 3), 'FaceColor', Colormap(Color_index(N), :), 'EdgeColor', 'none');
        end
    end

    % Define save path for the clustering figure
    figureSavePath = fullfile('C:\Users\Hashlu\Documents\MATLAB\REF\test', ...
                              participantID, sprintf('%s_clustering_figure.fig', participantID));
    saveas(gcf, figureSavePath);  % Save the figure

    
    %% Plot local efficiency 

% This section of the code is designed to visually represent the local
% efficency of the brain. its basically the same code just adjusting the
% parameter 

% first we will create a figure with the brain as we've done before

figure();
subplot(1,3,2);  
background_graph_Image; 
hold on; 

% Set an opacity level for visualization
alpha = .9;

% Calculate the maximum value from the local efficency matrix for normalization
aux_max = max(max(local_efficiency));

% Normalize the clustering values and scale them down based on alpha
x = local_efficiency - (alpha*aux_max);

% Compute the size of each node for plotting, using a sigmoid function for scaling
tam = 15./(1+exp(-4*(x)));

%Create a Colormap: the color of each ball has a linear dependency
% Generate a colormap using the 'jet' function which contains 1000 colors

Colormap = jet(1000); % Red-orange = higher clustering
                      % Blue-yellow = lower clustering

% Calculate scaling factors for the colormap index to ensure all values are within the valid range
a = 999/(max(tam)-min(tam));
b = 1 - a*min(tam);

% Compute the colormap index for each node based on its size
Color_index = round(a*tam + b);

% Generate a sphere that will represent each node in the plot
[x,y,z] = sphere(50,50); % adjust size here
% Loop over each node's coordinates to plot them
for N=1:size(CoorOpt_reg,1)
    
    % Only plot the node if it is not part of the SSlist (which might be a list of nodes to skip)
    if isempty(find(N==SSlist)==1)
        % Plot a colored sphere at each node's coordinates, scaled by 'tam' and colored based on 'Colormap'
        surf(tam(N)*x+CoorOpt_reg(N,1), tam(N)*y+CoorOpt_reg(N,2), ...
             tam(N)*z+CoorOpt_reg(N,3), 'FaceColor', ...
             Colormap(Color_index(N),:), 'EdgeColor', 'none');
    end
end

    % Define save path for the clustering figure
    figureSavePath = fullfile('C:\Users\Hashlu\Documents\MATLAB\REF\test', ...
                              participantID, sprintf('%s_localefficiency_figure.fig', participantID));
    saveas(gcf, figureSavePath);  % Save the figure

    
    