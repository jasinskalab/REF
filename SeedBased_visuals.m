function SeedBased_visuals(participantID, fwMCFilePath, SSlist, CorrelationCoefficient, Roi)
    % Visualize Seed-Based Connectivity
    % This function visualizes the connectivity based on the 
    % correlation matrix on a brain model for a given participant.
  
    % Parameters:
    % participantID - Participant identifer 
    % fwMCFilePath - Path to the participant's fwMC.mat file
    % SSlist - List of short channels
    % CorrelationCoefficient - Correlation matrix for the participant
    % Roi - seed
    
    %% Load in forward model
    data = load(fwMCFilePath, 'fwmodel');
    Adot = data.fwmodel.Adot(:,:,1);  % Load the sensitivity matrix
    mesh = data.fwmodel.mesh;  % Load mesh data

    % Take coordinates based on fwMC
    CoorOpt_reg = [];
    for nchn = 1:54  % 54 because we have 54 channels
        [~, index] = max(Adot(nchn, :));
        CoorOpt_reg = [CoorOpt_reg; mesh.vertices(index, :)];
    end

    % Extract directory from fwMCFilePath
    [saveDir, ~, ~] = fileparts(fwMCFilePath);
    
    % Extract the HbO correlation matrix 
    CorrelationMatrix = CorrelationCoefficient(:,:,3);

    % Define limits for the correlation values
    limit = [-1 1];

    % Set up the figure for brain visualization
    figure();
    subplot(1,3,2);
    background_graph_Image; % This function sets the background image of the brain
    set(gca,'CameraPosition',...
        [4465.88633611231 -122.094067886442 1263.15589092293],'CameraTarget',...
        [129.6960774264 138.177489128 141.54136287528],'CameraUpVector',...
        [-0.228800956760502 0.24649272775698 0.941749147782148],'CameraViewAngle',...
        1.85583529548837,'DataAspectRatio',[1 1 1],'LineStyleOrderIndex',50,...
        'PlotBoxAspectRatio',[1 1.18303201283536 1.08555630447355]);
    hold on;

    % Normalize and color map the correlation values
    corr_values = CorrelationMatrix(Roi,:);
    corr_values(corr_values > limit(2)) = limit(2);
    corr_values(corr_values < limit(1)) = limit(1);
    a = 999 / (limit(2) - limit(1));
    b = 1 - a * limit(1);
    Color_index = round(a * corr_values + b);
    Colormap = jet(1000);  % Create a colormap
    tam = 4;  % Node size
    [x, y, z] = sphere(50);  % Generate sphere coordinates for nodes

    % Plot each node on the brain model
    for N = 1:size(CoorOpt_reg, 1)
        if isempty(find(N == SSlist, 1))
            surf(tam * x + CoorOpt_reg(N, 1), tam * y + CoorOpt_reg(N, 2), tam * z + CoorOpt_reg(N, 3), 'FaceColor', Colormap(Color_index(N), :), 'EdgeColor', 'none');
            if N == Roi
                surf(tam * x + CoorOpt_reg(N, 1), tam * y + CoorOpt_reg(N, 2), tam * z + CoorOpt_reg(N, 3), 'FaceColor', [0 0 0], 'EdgeColor', 'none');  % Highlight ROI
            end
        end
    end
    % Optionally, save the figure
    savePath = fullfile(fileparts(fwMCFilePath), sprintf('%s_seedConnectivity.fig', participantID));
    saveas(gcf, savePath);  % Save the figure to file
end
