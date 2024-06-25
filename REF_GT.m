%% Working graph theory analytical pipeline - 25/06/2024

%% Caution 
            % ensure homer2 is installed
            % ensure Brain Connectivity Toolbox is installed
            % auxiliary scripts in path % adapted from WestNIRS conference
            % ensure no other toolboxes apart from homer2 is being called
            % on or in your path
           
            
%% Preprocessing : Organising and renaming resting state scans

% Directory that contains all the participant folders (after re_fNIRS_files.m)

% The reason we need to do this is because homer2 cannot read .nirs files

rootDir = 'C:\Users\Hashlu\Documents\MATLAB\REF\test2';

% Get a list of all subdirectories in the root directory
items = dir(rootDir);

% Filter out items that are not directories, and the '.' and '..' entries
subdirs = items([items.isdir] & ~ismember({items.name}, {'.', '..', '..'}));

% Loop through each subdirectory (participant folder)
for i = 1:length(subdirs)
    subdirName = subdirs(i).name;
    subdirPath = fullfile(rootDir, subdirName);
    
    % Check if the RS folder exists within the participant folder
    RSFolderPath = fullfile(subdirPath, 'RS');
    if exist(RSFolderPath, 'dir')
        % Check if there is a 'run' subdirectory within the RS folder
        runFolders = dir(fullfile(RSFolderPath, 'run*')); % Modify if 'run' has additional naming
        runFolders = runFolders([runFolders.isdir]); % Filter to include only directories
        
        for j = 1:length(runFolders)
            runFolderName = runFolders(j).name;
            runFolderPath = fullfile(RSFolderPath, runFolderName);
            
            % Find the .nirs file inside the 'run' directory
            nirsFiles = dir(fullfile(runFolderPath, '*.nirs'));
            
            % Check if there is at least one .nirs file
            if ~isempty(nirsFiles)
                % Assuming there is only one .nirs file inside the 'run' directory
                nirsFilePath = fullfile(runFolderPath, nirsFiles(1).name);
                
                % Generate the new filename based on the participant folder name
                newFileName = [subdirName, '_RS.mat'];
                newFilePath = fullfile(runFolderPath, newFileName);
                
                % Copy and rename the .nirs file
                copyfile(nirsFilePath, newFilePath);
                fprintf('Renamed %s to %s\n', nirsFiles(1).name, newFileName);
            else
                fprintf('No .nirs file found in %s\n', runFolderPath);
            end
        end
    else
        fprintf('No RS folder found for participant %s\n', subdirName);
    end
end

% Copy and rename all .nirs files to participant_RS.mat 

% Get a list of all items in the root directory
items = dir(rootDir);

% Filter out items that are not directories, and the '.' and '..' entries
subdirs = items([items.isdir] & ~ismember({items.name}, {'.', '..'}));

% Loop through each subdirectory
for i = 1:length(subdirs)
    subdirName = subdirs(i).name;
    subdirPath = fullfile(rootDir, subdirName);
    
    % Find all .nirs files in the current subdirectory
    nirsFiles = dir(fullfile(subdirPath, '*.nirs'));
    
    % Check if there is at least one .nirs file
    if ~isempty(nirsFiles)
        % Assuming there is only one .nirs file per subdirectory
        nirsFilePath = fullfile(subdirPath, nirsFiles(1).name);
        
        % Generate the new filename based on the participant folder name
        newFileName = [subdirName, '_RS.mat'];
        newFilePath = fullfile(subdirPath, newFileName);
        
        % Copy and rename the .nirs file
        copyfile(nirsFilePath, newFilePath);
    end
end

% Copy and rename all atlasViewer.mat to participant_fwMC.mat 

% Get a list of all items in the root directory
items = dir(rootDir);

% Filter out items that are not directories, and the '.' and '..' entries
subdirs = items([items.isdir] & ~ismember({items.name}, {'.', '..'}));

% Loop through each subdirectory (participant folder)
for i = 1:length(subdirs)
    subdirName = subdirs(i).name;
    subdirPath = fullfile(rootDir, subdirName);
    
    % Check if the RS folder exists within the participant folder
    RSFolderPath = fullfile(subdirPath, 'RS');
    if exist(RSFolderPath, 'dir')
        % Check if there are 'run' subdirectories within the RS folder
        runFolders = dir(fullfile(RSFolderPath, 'run*')); % Modify if 'run' has additional naming
        runFolders = runFolders([runFolders.isdir]); % Filter to include only directories
        
        for j = 1:length(runFolders)
            runFolderName = runFolders(j).name;
            runFolderPath = fullfile(RSFolderPath, runFolderName);
            
            % Find the atlasViewer.mat file inside the 'run' directory
            atlasViewerFiles = dir(fullfile(runFolderPath, 'atlasViewer.mat'));
            
            % Check if there is at least one atlasViewer.mat file
            if ~isempty(atlasViewerFiles)
                % There should only be one atlasViewer.mat file per 'run' directory
                atlasViewerFilePath = fullfile(runFolderPath, atlasViewerFiles(1).name);
                
                % Generate the new filename based on the participant folder name
                newFileName = [subdirName, '_fwMC.mat'];
                newFilePath = fullfile(runFolderPath, newFileName);
                
                % Copy and rename the atlasViewer.mat file
                copyfile(atlasViewerFilePath, newFilePath);
                fprintf('Renamed %s to %s in %s\n', atlasViewerFiles(1).name, newFileName, runFolderPath);
            else
                fprintf('No atlasViewer.mat file found in %s\n', runFolderPath);
            end
        end
    else
        fprintf('No RS folder found for participant %s\n', subdirName);
    end
end

%% Automated Analysis and Export for Each _RS.mat 

matFiles = dir(fullfile(rootDir, '**', 'run', '*_RS.mat')); % Find all _RS.mat files in 'run' directories

% Initialize an empty table to store results

resultTable = table('Size', [0, 7], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'double',}, ...
    'VariableNames', {'participantID', 'mean_clustering', 'path_length', 'small_worldness', 'SWP', 'modularity', 'global_efficiency'});



for i = 1:length(matFiles)
    matFilePath = fullfile(matFiles(i).folder, matFiles(i).name); % Full path to the _RS.mat file
    [~, fileName, ~] = fileparts(matFilePath); % Extract participant name from file name

    % Extract the base participant ID without '_RS'
    participantID = regexprep(fileName, '_RS$', ''); % Remove '_RS' at the end of the fileName
    
    % Load the _RS.mat file
    load(matFilePath); % Assuming the data variable names inside are consistent
    
%% List of short channels for the used probe :
     % go through SD.MeasList and locate the position of detectors & short channel
     % short channels = cell with detectors that exceed number of channels
     % (>16)

     % this list should be consistent with our montage but it's worth going through SD.MeasList to confirm

    % SSlist = [ 12 16 25 28 35 46 50];

% List of short channels for the used probe (automated)

% First we need to identify the number of channels we used. This can be 
% determined by reading SD.MeasList. 

                % Column 1 refers to S/D number
                % Column 2 refers to S/D (or short channel) used with S/D 
                % to form a Data Point

% therefore we need to find the highest number in Column 1 to determine
% Number of channels we used per S/D

NumOptodes = max(SD.MeasList(:, 1));

% Determine number of data points we have in the given montage. All the 
% data points are located within the SD.MeasList therefore we need to find
% it's length. As half of these datapoints are sources and the other half 
% are detectors, we will only require half as we only have one set of 
% short channels - not 2

% Data points = S + D (or D + S) pair used to collect a reading
% we are only focusing on the first column as this doesn't collect SC 
% data points

NumChannels = (size(SD.MeasList, 1)) / 2 ;

% The second column in SD.MeasList contains SC information, these will take
% on channel numbers greater than our number of channels used therefore
% we will find these and put them in a list

ShortChannels = find(SD.MeasList(1: NumChannels, 2) > NumOptodes) ;

% Now we reshape the data type into a List for the rest of the analysis

SSlist = reshape(ShortChannels, 1, []) ;

%% Find channels with low SNR (signal-to-noise ratio) - work in progress

    %   Low = weak signal
    %   High = strong signal

    % Adapted using Homer 
    
    % Threshold for deciding between bad and good channels
    SNR_threshold = 8;

    % Remove long drifts from the data that can be misleading when computing
    % the quality of the channel
  
    Baseline = mean(d);
    d = detrend(d)+Baseline;

  SD = enPruneChannels(d,SD,ones(size(d,1),1),...
          [-10 10^7],SNR_threshold,[0 100],0);
        
     % need to determine dRange

    BadChannels = find(SD.MeasListActAuto==0);


%% Compute Optical Density
dOD = hmrIntensity2OD(d);

% Hard coded for script to work

% F = sampling rate, which for our study is 5.1Hz

SD.f = 5.1 ;

%% Motion Correction

% Spline interpolation followed by wavelet decomposition
dOD = Hybrid_motion_correction(dOD,SD);

%% Compute Hemoglobin Concentration changes
dc = hmrOD2Conc...
    (dOD, SD, [6 6]); 

% Permute dc
dc = permute(dc,[1 3 2]);

% Band-Pass Filter Hemoglobin concentrations
dc = hmrBandpassFilt...
    (dc, SD.f, 0.009, 0.08);

%% Short-Channel Regression 

 AdditionalRegressors = [];


for Hb=1:2
    
    for Nchan=1:size(dc,2)
        
        % Step 1: Perform Regression in a given channel Nchan
        y = dc(:,Nchan,Hb);
        
        % Step 2: Create Design Matrix (X) for regression
        X=[];
        
        if ~isempty(SSlist)
            % Add HbO and HbR in the design matrix
            Xshort = [dc(:,SSlist,1),dc(:,SSlist,2)];
            
            % PCA for removing collinearity
            [coeff_pca Xshort_pca] = pca(Xshort);
            
            % Update Design Matrix with the PCs
            X = [X,Xshort_pca];
        end   
        
        if ~isempty(AdditionalRegressors)
            
            % Perform shift in each additional regressor
            % that maximizes the correlation with the fNIRS channel.
            % The maximum allowed shift is 20 seconds.
            
            maxLag = round(20*SD.f);
         %   [y,AdditionalRegressors_s,shift_AD,coor_max] = ...
        %        AdjustTemporalShift_fnirs_course...
       %         (y,AdditionalRegressors,maxLag);

    for Nadd = 1:size(X,2)
    
    % Compute xcorr to find shifts between the physiology
    [corrValues Lags] = ...
        xcorr(y,X(:,Nadd),maxLag,'coeff');
    
    [max_coor_value index_Lag_corr]  = max(abs(corrValues));
    
    shift(Nadd) = Lags(min(index_Lag_corr));
    coor_max(:,Nadd) = corrValues(index_Lag_corr);

    % Shifted Additional Regressors
    X_new(:,Nadd) =...
        circshift(X(:,Nadd),shift(Nadd));
    
    end

% Correct Design Matrix and the Y vector
% by removing the firts and last points based on the maximum allowed LAG

y_new=y;

X_new(1:maxLag,:) = [];
y_new(1:maxLag,:) = [];

X_new(end-maxLag:end,:) = [];
y_new(end-maxLag:end,:) = [];

            % We have to cut the begning and end of the design matrix
            % as we did with the shifted time series and y vector inside
            % "AdjustTemporalShift_for_Regression".
            if ~isempty(X)
                X(1:maxLag,:) = [];
                X(end-maxLag:end,:) = [];
            end
            
            % Next, we add the shifted additional regressors
            % to the design matrix
            X = [X,X_new];
        end
        
    
 [Dummy, StatsDummy] = robustfit(X,y,'bisquare',[],'on');
        
        % Save filtered data (residual)
        filtered_dc(:,Nchan,Hb) = StatsDummy.resid;
        
        % Save Additional Regressors Shifts for further analysis
        if ~isempty(AdditionalRegressors)
            StatsDummy.shift_AD = shift_AD./SD.f;
            StatsDummy.coor_AD = coor_max;
        end
        
        % Save Stats for further analysis
        Stats{Nchan}{Hb} = StatsDummy;
        
        clear y X StatsDummy;
        
        
    end
    
    
end


%% Compute total hemoglobin for filtered data (no short channels)
filtered_dc(:,:,3) = filtered_dc(:,:,1) + filtered_dc(:,:,2);


%% Correlation 


%% Remove autocorrelation 

Pmax = round(20*SD.f);

% Whitened data
dc_w = nan*zeros(size(filtered_dc));


% Time Series Length
n = size(filtered_dc,1);


% Run on HbO and HbR
for Hb=1:2
    
    for Nchannel=1:size(filtered_dc,2)
        
        if isempty(find(isnan(filtered_dc(:,Nchannel,Hb))==1))
            
            clear y yf a vt
            
            % Get Original Time Series
            y = filtered_dc(:,Nchannel,Hb);
            
            for P=1:Pmax
                % For a given parameters P we find the coefficients that
                % minimize autoregressive model (AR(P));
                a = aryule(y,P);
                
                % Once we have the parameters a, we can filter the error
                % to find the new non atucorrelated error (vt).
                vt = filter(a,+1,y);
                
                % Next, we can compute the baysian information
                % criterion (BIC(P)).
                
                % Log Likelihood
                LL = -1*(n/2)*log( 2*pi*mean(vt.^2))+...
                    -0.5*(1/mean(vt.^2))*sum(vt.^2);
                
                % Baysian information
                BIC(P) = -2*LL+P*log(n);
            end
            
            %Optimal is the P that minimizes BIC
            [~,OptimalP] = min(BIC);
            
            AR_Parameters = aryule(y,OptimalP); %Find parameters
            
            % Filter y
            yf = filter(AR_Parameters,+1,y);
            
            % Update dc_w
            dc_w(:,Nchannel,Hb) = yf;
            
            % Save OptimalP for double checking
            SD.Optimal_P(Nchannel,Hb) = OptimalP;
            
        end
        
    end
    
end


%% Compute Total Hemoglobin for whitened data 

%Hbt           %HbO            %HbR
dc_w(:,:,3) = dc_w(:,:,1) + dc_w(:,:,2);

%% Remove undetermined points (first P_max points)
dc_w = dc_w(Pmax+1:end,:,:);

    
%% Compute Pearson Correlation Coefficient (regressed Short Channels)

     % Exclude channels from Correlation Matrix
    
     exclude_channels = unique([SSlist,BadChannels']);
    
     % Compute for HbO, HbR, and HbT  
    
    for Hb=1:3
        
        CorrelationCoefficient(:,:,Hb) = ...
            corrcoef(dc_w(:,:,Hb)); 
    end

     % Assign "Exclude channels" as nan
%     CorrelationCoefficient(exclude_channels,:,:) = nan;
%     CorrelationCoefficient(:,exclude_channels,:) = nan;
     
    % Remove "Exclude channels"
%     CorrelationCoefficient(exclude_channels,:,:) = [];
%     CorrelationCoefficient(:,exclude_channels,:) = [];
    
       % Assign "Exclude channels" as zeros
     CorrelationCoefficient(exclude_channels,:,:) = 0;
     CorrelationCoefficient(:,exclude_channels,:) = 0; 

      
          
%% Graph theory

% Define a threshold, for graph theoretical analysis, 
% it is often necessary to convert these continuous values into a 
% binary format (1s and 0s) where 1 indicates a connection and 0 indicates 
% no connection. This is also necessary for the production of the adjacency
% matrix

threshold = 0.3;

% 0.3 was selected as it was recommended in the fNIRS course

%% Graph theory : Preprocessing - Adjacency Matrix

% An adjacency matrix provides us a compact way to represetn the structure
% of anetwork. Each element of the matrix will indicate whether a pair of
% nodes are connected

% We must also define what particular Hb type do we care to use for our
% correalation analysis/graph theoretical measures

% 1 - HbO, 2 - HbR, and 3 - HbT.  
ChosenHb = 1;

CorrMatrix = CorrelationCoefficient(:,:,ChosenHb);

A = CorrMatrix;

for m=1:size(A,1)
    A(m,m)=0;
end

for i = 1:size(A,1)
    for j = 1:size(A,1)
        if A(i,j) >= 0.3
            A(i,j) = 1;
        else
            A(i,j) = 0;
        end
    end
end

% Assign short channel enties as nan
A(SSlist,:) = nan;
A(:,SSlist) = nan;

K1 = nansum(A);


%% Graph Theory : Network Properties

    %   Local and Global efficiency
    %   Clustering Coefficient

% Compute Node degree
degree = nansum(A,2);

% Clustering coefficient

% Clustering coefficient matrix

B = A; % sets a separete variable as the adjacency matrix to ensure A 
        % doesn't get overwritten

Clustering = nan*zeros(length(B),1);

% Remove SSlist entries
chan_index = 1:1:length(B);
chan_index(SSlist) = [];

B(SSlist,:) = [];
B(:,SSlist) = [];

n=length(B);
C=zeros(n,1);

for u=1:n
    V=find(B(u,:));
    k=length(V);
    if k>=2;                %degree must be at least 2
        S=B(V,V);
        C(u)=sum(S(:))/(k^2-k);
    end
end

cnt = 0;
for Nchan = chan_index
   
   cnt = cnt+1;
   Clustering(Nchan) = C(cnt);       
    
end

% Local efficiency
[local_efficiency] = ...
    efficiency_bin_REF(A,1,SSlist);


% Global efficiency
[global_efficiency] = ...
   efficiency_bin_REF(A,0,SSlist);

% Path length

% Calculate path lengths
path_length = charpath(distance_bin(A));  
% This is the average shortest path length between all pairs of nodes and 
% provides a measure of the overall 'closeness' of the network.


%% Graph Theory : Small worldness

% Calculating small-worldness involves comparing the clustering coefficient 
% and characteristic path length of your network to those of a corresponding 
% random graph. Small-world networks are characterized by a high clustering 
% coefficient relative to random networks and a short path length close to 
% that of random networks.

% we already have calculated clustering coefficient and characteristic path length
% so we will now prepare  the random graph

num_edges = nnz(A) / 2;  % Number of edges in the original graph
N = size(A, 1);  % Number of nodes
P = num_edges / (N * (N - 1) / 2);  % Connection probability
A_random = rand(N) < P;
A_random = triu(A_random, 1);
A_random = A_random + A_random';  % Make the matrix symmetric


% Now we will run the same network property calculations for A_random this
% is done so we can have an actual comparison we will just copy the code
% above and paste it here, adjusting variable names


B_random = A_random; % sets a separete variable as the adjacency matrix to ensure A 
        % doesn't get overwritten

Clustering_random = nan*zeros(length(B_random),1);

% Remove SSlist entries
chan_index = 1:1:length(B_random);
chan_index(SSlist) = [];

B_random(SSlist,:) = [];
B_random(:,SSlist) = [];

n=length(B_random);
C=zeros(n,1);

for u=1:n
    V=find(B_random(u,:));
    k=length(V);
    if k>=2;               
        S=B_random(V,V);
        C(u)=sum(S(:))/(k^2-k);
    end
end

cnt = 0;
for Nchan = chan_index
   
   cnt = cnt+1;
   Clustering_random(Nchan) = C(cnt);     
end

% Now we will calculate the characteristic path length for the random graph
% we will also just copy the same code over

path_length_random = charpath(distance_bin(A_random));


% Calculate small-worldness

% Small-worldness(sigma) is defined as the ratio of relative clustering
% coefficent (gamma) and relative path length (lambda) between the network 
% of interest and random network

% a network is considered small world if lamba > 1


% filter out NaN values from both clustering and clustering_random
valid_indices = ~isnan(Clustering) & ~isnan(Clustering_random);
filtered_clustering = Clustering(valid_indices);
filtered_clustering_random = Clustering_random(valid_indices);

% Compute the means of the filtered clustering coefficients
mean_clustering = mean(filtered_clustering);
mean_clustering_random = mean(filtered_clustering_random);

% Compute gamma, lambda, small_wordlness (sigma)
gamma = mean_clustering / mean_clustering_random;
lambda = path_length / path_length_random;
small_worldness = gamma / lambda;

% Small world propensitiy

% Small-world propensity is a relatively newer metric designed to assess the 
% small-world characteristics of a network while addressing some limitations 
% in the traditional sigma metric, especially when used with brain networks 
% or other complex structures. This metric is designed to be more robust against
% variations in network size, density, and degree distribution, which can affect 
% the comparability of the traditional small-worldness measure.

% Calculate DeltaC: Normalized Clustering Coefficient Deviation
delta_C = (mean_clustering - mean_clustering_random) / ...
          (max([Clustering; Clustering_random]) - mean_clustering_random);
delta_C = max(0, min(delta_C, 1)); 

% Calculate DeltaL: Normalized Path Length Deviation
delta_L = (path_length_random - path_length) / ...
          (path_length_random - min([path_length; path_length_random]));
delta_L = max(0, min(delta_L, 1));  

% Calculate Small World Propensity using ?C and ?L
SWP = 1 - sqrt(delta_C^2 + delta_L^2) / sqrt(2)

%% Graph Theory : Modularity

% Quantifies the degree to which a network can be divided into clearly
% defined groups or communities of nodes. They are defined by dense
% connections withhin themelves or sparse connections between different
% communities 


% Using BCT we can obtain the modularity score (Q) which measures the
% strength of the division of a network into modules - closer to 1
% indicates more densely connected to each other

% We also obtain Ci, which will provide an array where each element
% corresponds to a node in the network - and the value at each index
% indicates the community to which the node has been assigned to


% Create a new matrix for modularity calculation based on A
A_m = A;

% Replace NaN values in A_m assuming NaN signifies no connection
A_m(isnan(A_m)) = 0;

% Create a logical index that excludes short channels
indices = true(size(A_m, 1), 1);  % Start with all indices included
indices(SSlist) = false;          % Exclude short channels

% Remove rows and columns corresponding to short channels
A_m = A_m(indices, indices);

% Calculate Modularity

[Ci, Q] = modularity_und(A_m);
   
modularity = Q;


%% Figures - Image Reconstruction

% Load in forward model        
        
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

 fwMCFilePath = fullfile(rootDir, subdirName, 'RS', 'run', strcat(subdirName, '_fwMC.mat'));

% Image reconstruction 
    GT_visuals(A, fwMCFilePath, participantID, SSlist, Clustering, local_efficiency);


%% Save results into a temporary table


% Ensure all variables are appropriate for insertion into a table
        participantID = string(participantID); 
        mean_clustering = double(mean_clustering);  
        path_length = double(path_length);  
        small_worldness = double(small_worldness);  
        SWP = double(SWP); 
        modularity = double(modularity); 
        global_efficiency = double(global_efficiency); 

% Create a temporary table for this iteration's data
tempTable = table(participantID, mean_clustering, path_length, small_worldness, SWP, modularity, global_efficiency, ...
    'VariableNames', {'participantID', 'mean_clustering', 'path_length', 'small_worldness', 'SWP', 'modularity', 'global_efficiency'});

% Append the temporary table to the results table
resultTable = [resultTable; tempTable];

    clear filtered_dc SWP small_worldness
end

% Export resultsTable to CSV

writetable(resultTable, 'GT_results_test.csv');


