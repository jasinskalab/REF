%% working functional connectivity pipeline --- 07/04/24

% Caution 
            % ensure homer2 is installed
            % auxiliary scripts in path % adapted from WestNIRS conference
            % ensure no other toolboxes apart from homer2 is being called
            % on or in your path

%% Preprocessing %%

% We currently do not have any physiological measures. much of the
% preprocessing involves preparing variables and data for analysis along
% with regressing short channels out 


%% Copy and rename all .nirs files to participant_RS.mat
% Directory that contains all the participant folders (after re_fNIRS_files.m)

% The reason we need to do this is because homer2 cannot read .nirs files

rootDir = 'C:\Users\Hashlu\Documents\MATLAB\REF\re-org';

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
%% Copy and rename all .nirs files to participant_RS.mat and also collect .pos and get _origins and _others.csv

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
    
      % Find all .pos files in the current subdirectory
    posFiles = dir(fullfile(subdirPath, '*.pos'));
    
    for j = 1:length(posFiles)
        % Skip hidden macOS metadata files that start with '._'
        if startsWith(posFiles(j).name, '._')
            continue;  % Skip this file and move to the next one
        end
        
        inputFilePath = fullfile(subdirPath, posFiles(j).name);

% Initialize lineNumber before starting to read the file
lineNumber = 0;

% Open the input .pos file
fileID = fopen(inputFilePath, 'r');
assert(fileID > 0, ['Failed to open file: ', inputFilePath]);  % debugging message
 % Determine the multiplier for X coordinate based on LPA's X coordinate
        coordinateMultiplier = 10; % Default multiplier for X coordinate
        while ~feof(fileID)
            line = fgetl(fileID);
            if contains(line, 'LPA')
                data = strsplit(line, '\t');
                % Check the sign of the X coordinate before transformation 
                if str2double(data{3}) > 0
                    coordinateMultiplier = -10; % Adjust multiplier for X if LPA's X is positive
                end
                break; % LPA found, no need to continue
            end
        end
        
        % Reset to the beginning of the file after scanning for LPA
        fseek(fileID, 0, 'bof');
        % Initialize an empty array to store the transformed data

dataTransformed = {};
lineCount = 0;  % Keep track of the total number of lines read

% Read the file line by line
while ~feof(fileID)
    line = fgetl(fileID);
    if isempty(line) || ~ischar(line)  % Skip empty lines or non-character data
        continue;
    end
    data = strsplit(line, '\t');
    
    % For lines with initial number, Label, X, Y, Z (lines 1 to 36)
    if length(data) == 5  
        optode = data{2};  % Label
        x = str2double(data{4})* coordinateMultiplier;  % Original Y position, now X
        y = str2double(data{3})* 10;  % Original X position, now Y
        z = str2double(data{5})* 10;  % Z remains the same
        
        % Append to dataTransformed
        dataTransformed{end+1, 1} = optode;
        dataTransformed{end, 2} = x;
        dataTransformed{end, 3} = y;
        dataTransformed{end, 4} = z;

        % Debugging print statement to confirm data addition
        disp(['Added data for line ', num2str(lineNumber), ': ', optode, ', X: ', num2str(x), ', Y: ', num2str(y), ', Z: ', num2str(z)]);
        
    % For lines without initial number, directly Label, Y, X, Z (lines 37 to 42)
    elseif length(data) == 4  
        optode = data{1};  % Label
        x = str2double(data{3})* coordinateMultiplier;  % Original Y position, now X
        y = str2double(data{2})* 10;  % Original X position, now Y
        z = str2double(data{4})* 10;  % Z remains the same
        
        % Append to dataTransformed
        dataTransformed{end+1, 1} = optode;
        dataTransformed{end, 2} = x;
        dataTransformed{end, 3} = y;
        dataTransformed{end, 4} = z;

        % Debugging print statement to confirm data addition
        disp(['Added data for line ', num2str(lineNumber), ': ', optode, ', X: ', num2str(x), ', Y: ', num2str(y), ', Z: ', num2str(z)]);
    end
    lineNumber = lineNumber + 1; % Increment line number for each processed line
end

disp(['Total lines read: ', num2str(lineCount)]);  % Display total lines read from the file

% Close the input file
fclose(fileID);

disp(['Size of dataTransformed: ', num2str(size(dataTransformed))]);

% Convert the cell array to a table
T = cell2table(dataTransformed, 'VariableNames', {'Label', 'X', 'Y', 'Z'});

if size(T, 1) >= 41
    originsData = T(32:41, :);
else
    warning('T does not contain enough rows to extract originsData.');
    % Handle the case accordingly, perhaps by skipping the extraction or adjusting the range
end

if size(T, 1) >= 32
    % Extract from row 32 to the end of the table
    originsData = T(32:end, :);
else
    warning('T does not contain enough rows to extract expected originsData.');
    % Optionally handle this scenario, e.g., by creating an empty table for originsData
    originsData = table([], [], [], [], 'VariableNames', {'Label', 'X', 'Y', 'Z'});
end

% Convert names of fiducials to match nfri anchor names
originsData.Label = strrep(originsData.Label, 'LPA', 'ALHS');
originsData.Label = strrep(originsData.Label, 'RPA', 'ARHS');
originsData.Label = strrep(originsData.Label, 'NA', 'NzHS');


% Table for channel data for others file
channelData = table({}, zeros(0), zeros(0), zeros(0), 'VariableNames', {'Label', 'X', 'Y', 'Z'});

optodePairs = {};

% Fill the array step by step with  optode pairs this is hard coded
% based on REF montage, working on automating this based on SD.MeasList
 
optodePairs{end+1} = {'CH01', 'S01', 'D01'};
optodePairs{end+1} = {'CH02', 'S01', 'D02'};
optodePairs{end+1} = {'CH03', 'S02', 'D01'};
optodePairs{end+1} = {'CH04', 'S02', 'D02'};
optodePairs{end+1} = {'CH05', 'S02', 'D03'};
optodePairs{end+1} = {'CH07', 'S03', 'D01'};
optodePairs{end+1} = {'CH08', 'S03', 'D03'};
optodePairs{end+1} = {'CH09', 'S04', 'D02'};
optodePairs{end+1} = {'CH10', 'S04', 'D03'};
optodePairs{end+1} = {'CH11', 'S04', 'D04'};
optodePairs{end+1} = {'CH13', 'S05', 'D03'};
optodePairs{end+1} = {'CH14', 'S05', 'D04'};
optodePairs{end+1} = {'CH15', 'S05', 'D05'};
optodePairs{end+1} = {'CH17', 'S06', 'D05'};
optodePairs{end+1} = {'CH18', 'S06', 'D06'};
optodePairs{end+1} = {'CH19', 'S07', 'D04'};
optodePairs{end+1} = {'CH20', 'S07', 'D05'};
optodePairs{end+1} = {'CH21', 'S07', 'D07'};
optodePairs{end+1} = {'CH22', 'S08', 'D05'};
optodePairs{end+1} = {'CH23', 'S08', 'D06'};
optodePairs{end+1} = {'CH24', 'S08', 'D07'};
optodePairs{end+1} = {'CH26', 'S09', 'D08'};
optodePairs{end+1} = {'CH27', 'S09', 'D10'};
optodePairs{end+1} = {'CH29', 'S10', 'D08'};
optodePairs{end+1} = {'CH30', 'S10', 'D09'};
optodePairs{end+1} = {'CH31', 'S11', 'D08'};
optodePairs{end+1} = {'CH32', 'S11', 'D09'};
optodePairs{end+1} = {'CH33', 'S11', 'D10'};
optodePairs{end+1} = {'CH34', 'S11', 'D11'};
optodePairs{end+1} = {'CH37', 'S12', 'D11'};
optodePairs{end+1} = {'CH38', 'S12', 'D13'};
optodePairs{end+1} = {'CH39', 'S13', 'D09'};
optodePairs{end+1} = {'CH40', 'S13', 'D11'};
optodePairs{end+1} = {'CH41', 'S13', 'D12'};
optodePairs{end+1} = {'CH42', 'S14', 'D11'};
optodePairs{end+1} = {'CH43', 'S14', 'D12'};
optodePairs{end+1} = {'CH45', 'S14', 'D14'};
optodePairs{end+1} = {'CH47', 'S15', 'D12'};
optodePairs{end+1} = {'CH48', 'S15', 'D14'};
optodePairs{end+1} = {'CH49', 'S15', 'D15'};
optodePairs{end+1} = {'CH51', 'S16', 'D01'};
optodePairs{end+1} = {'CH52', 'S16', 'D13'};
optodePairs{end+1} = {'CH53', 'S16', 'D14'};
optodePairs{end+1} = {'CH54', 'S16', 'D15'};

% Loop through each  optode pair for channel calculation
for i = 1:length(optodePairs)
    channelLabel = optodePairs{i}{1}; % Channel label
    optode1 = optodePairs{i}{2}; % First optode in the pair
    optode2 = optodePairs{i}{3}; % Second optode in the pair
    
    % Find indices for both optodes in the pair
    idx1 = find(strcmp(T.Label, optode1));
    idx2 = find(strcmp(T.Label, optode2));
    
    if ~isempty(idx1) && ~isempty(idx2)
        % Calculate mean coordinates for the exact pair
        meanX = mean([T.X(idx1); T.X(idx2)]);
        meanY = mean([T.Y(idx1); T.Y(idx2)]);
        meanZ = mean([T.Z(idx1); T.Z(idx2)]);
        
        % Create a newRow table with matching column names
        newRow = table({channelLabel}, meanX, meanY, meanZ, 'VariableNames', {'Label', 'X', 'Y', 'Z'});

        % Check if channelData is empty by examining its height
        if height(channelData) == 0
            channelData = newRow; % If channelData is empty, initialize it with newRow
        else
            channelData = [channelData; newRow]; % Otherwise, append newRow to channelData
        end
    else
        fprintf('Optode pair not found: %s and %s, for channel %s\n', optode1, optode2, channelLabel);
    end
end
% Combine _others data with channelData in a table - othersData
othersData = [T(1:31, :); channelData];

        originsOutputFilePath = fullfile(subdirPath, strrep(posFiles(j).name, '.pos', '_origins.csv'));
        othersOutputFilePath = fullfile(subdirPath, strrep(posFiles(j).name, '.pos', '_others.csv'));
        writetable(originsData, originsOutputFilePath, 'WriteVariableNames', true);
        writetable(othersData, othersOutputFilePath, 'WriteVariableNames', false);
    end
end



%% Automated Analysis and Export for Each _RS.mat File and _con.CSV 

matFiles = dir(fullfile(rootDir, '**', 'run', '*_RS.mat')); % Find all _RS.mat files in 'run' directories

for i = 1:length(matFiles)
    matFilePath = fullfile(matFiles(i).folder, matFiles(i).name); % Full path to the _RS.mat file
    [pathStr, fileName, ~] = fileparts(matFilePath); % Extract participant name from file name and save to _RS
    
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

%% Drawing figures
    
    figure('Renderer', 'painters', 'Position', [50 100 1200 300])
    subplot(1,3,1)
    imagesc(CorrelationCoefficient(:,:,1),[-1 1]);
    colormap jet
    title('HbO - Only SC')
    
    subplot(1,3,2)
    imagesc(CorrelationCoefficient(:,:,2),[-1 1]);
    colormap jet
    title('HbR - Only SC');
    
    subplot(1,3,3)
    imagesc(CorrelationCoefficient(:,:,3),[-1 1]);
    colormap jet
    title('HbT - Only SC')
    


%% Fisher transformation to Z score

z_score = atanh(CorrelationCoefficient)

% Adjust (:,:,n) where n = 1:3 = HbO, HbR, HbT

G = graph(z_score(:,:,1))



%% Export Graph to CSV

% We are interested in the G.Edges - this contains the respective Z scores
% between channel pair

%Write G.Edges to table and split the nested table to 3 separate columns 

correlation_table = splitvars(G.Edges)
 
  %% Export Results to CSV
    % Assuming 'correlation_table' is the result table you want to export
    csvFileName = sprintf('%s_con.csv', fileName); % Name of the output CSV file
    csvFilePath = fullfile(pathStr, csvFileName); % Full path to the output CSV file
    
    % Save the table to CSV
    writetable(correlation_table, csvFilePath);
    
    clear filtered_dc
    
    % Print a message to indicate completion
    fprintf('Analysis and export completed for %s\n. File saved to: %s\n', fileName, csvFilePath);
 

end

%% Image Reconstruction (Seed-Based Analysis) - work in progress

% Ensure Forward Model for participant is loaded (create using AtlasViewer)

%load fwMC.mat


%% Graph theoretical network analysis - work in progress 


