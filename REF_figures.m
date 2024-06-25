%% working image reconstruciton pipeline --- 20/06/24

        % Adapted using WestNIRs course
        % Obtain forward models using Homer2 (refer to REF_GT/REF_FC)
        

%% Load in forward model        
        
% the forward model is generated using homer2 atlasviwer, The forward model 
% predicts the expected light transport through the head given certain 
% properties of the tissue this model is based on the optical properties 
% of the tissues (such as scattering and absorption coefficients) and 
% the geometry of the head and the placement of fNIRS sensors and sources. 
% The outputs of the forward model are typically light intensity or photon 
% density distributions, which are used to infer the changes in oxygenated 
% and deoxygenated hemoglobin concentrations from the measured light attenuations in the brain.


% The fwMC file has a bunch of different variables, the only two we are
% interested in are, Adot and mesh - Adot tells us the sensitivity matrix
% and mesh refers to the volume in which the the photons are migrating - in
% essence the space we capture, or in other wrods the structure of the head

fwMCFilePath = "C:\Users\Hashlu\Documents\MATLAB\REF\test2\REF_005\RS\run\REF_005_fwMC.mat"
data = load(fwMCFilePath, 'fwmodel');
 
% load in specifically Adot and mesh

Adot = data.fwmodel.Adot(:,:,1); % the Adot file has 2 matrices, we are just
                                 % interested in the first one                  
mesh = data.fwmodel.mesh; % which contains faces and vertices variable

% Take coordinates based on fwMC
CoorOpt_reg = [];
for nchn = 1:54 % 54 becasue we have 54 channels 
    [~,index] = max(Adot(nchn,:));
    CoorOpt_reg = [CoorOpt_reg;mesh.vertices(index,:)];
end


%% Plot links and nodes on brain - Graph Theory

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

participantID = 'REF_005'; % Define or fetch the participant ID dynamically if needed
figureSavePath = fullfile('C:\Users\Hashlu\Documents\MATLAB\REF\test\REF_005\RS\run', sprintf('%s_linknodes_figure.fig', participantID));

% Save the figure to the specified path
saveas(gcf, figureSavePath);


%% Plot clustering coefficient on the brain - Graph Theory

% This section of the code is designed to visually represent the clustering coefficients
% of nodes distributed across a brain

% first we will create a figure with the brain as we've done before

figure();
subplot(1,3,2);  
background_graph_Image; 
hold on; 

% Set an opacity level for visualization
alpha = .9;

% Calculate the maximum value from the Clustering matrix for normalization
aux_max = max(max(Clustering));

% Normalize the clustering values and scale them down based on alpha
x = Clustering - (alpha*aux_max);

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

participantID = 'REF_005'; % Define or fetch the participant ID dynamically if needed
figureSavePath = fullfile('C:\Users\Hashlu\Documents\MATLAB\REF\test\REF_005\RS\run', sprintf('%s_clustering_figure.fig', participantID));

% Save the figure to the specified path
saveas(gcf, figureSavePath);


%% Plot local efficiency - Graph Theory

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


participantID = 'REF_005'; % Define or fetch the participant ID dynamically if needed
figureSavePath = fullfile('C:\Users\Hashlu\Documents\MATLAB\REF\test\REF_005\RS\run', sprintf('%s_localefficiency_figure.fig', participantID));

% Save the figure to the specified path
saveas(gcf, figureSavePath);

%% Seed-Based Connectivity  

% This section will outline the process of plotting the calculated
% correlation matrix onto a brain model, visualizing seed and the
% surounding correlations between other channels

% First we will need to define subset the HbO correlation matrix

CorrelationMatrix = (CorrelationCoefficient(:,:,3));

% Now we will define our seed (as Roi) refer to your montage to select
% your seed - i.e, channel.

Roi = 31; % L-STG


% Next we will define a variable to collect all the correlation values
% associated between the ROI across all channels

corr_values = CorrelationMatrix(Roi,:);

% Define limits for the corrlations values (this will allow us to
% standardize the colour mapping)

limit = [-1 1]

% Create a figure with the brain as we've done before

figure()
subplot(1,3,2)
background_graph_Image;
set(gca,'CameraPosition',...
    [4465.88633611231 -122.094067886442 1263.15589092293],'CameraTarget',...
    [129.6960774264 138.177489128 141.54136287528],'CameraUpVector',...
    [-0.228800956760502 0.24649272775698 0.941749147782148],'CameraViewAngle',...
    1.85583529548837,'DataAspectRatio',[1 1 1],'LineStyleOrderIndex',50,...
    'PlotBoxAspectRatio',[1 1.18303201283536 1.08555630447355]);
hold on

%%% Create a Colormap
Colormap = jet(1000);

% Set the correlation values within the defined limits to ensure colour
% mapping isn't skewed
corr_values(corr_values>limit(2)) = limit(2);
corr_values(corr_values<limit(1)) = limit(1);

% Scale and translate the correlaton values to map them to the color map

a = 999/(limit(2)-limit(1));
b = 1 - a*limit(1);

Color_index = round(a*corr_values +b);

% Define node size to be plotted on brain

tam = 4;
[x,y,z] = sphere(50,50);

% Iterate over all channels to plot each node

for N=1:size(CoorOpt_reg,1)
    
    if isempty(find(N==SSlist)==1) % checks for short channels
        surf(tam*x+CoorOpt_reg(N,1),tam*y+CoorOpt_reg(N,2),...
            tam*z+CoorOpt_reg(N,3),'FaceColor',...
            Colormap(Color_index(N),:),'EdgeColor','none');
        
        % Roi will be black to contrast better
        if N==Roi
           surf(tam*x+CoorOpt_reg(N,1),tam*y+CoorOpt_reg(N,2),...
            tam*z+CoorOpt_reg(N,3),'FaceColor',...
            [0 0 0],'EdgeColor','none');
            
        end
    end
end


















