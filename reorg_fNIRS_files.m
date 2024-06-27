function reorg_fNIRS_files()
% Reorganize fNIRS files into a new folder structure based on participant and experiment details.
    % Originally created using python, adapted from re_org.py


%% GUI

% Just for ease of use I have made a GUI

% Select the root directory containing the original files.
rootDirectory = uigetdir('', 'Select Root Directory');
if rootDirectory == 0
    return;
end

% Select the new participant directory where files will be reorganized and copied.
newParticipantDirectory = uigetdir('', 'Select Target Directory for New Organization');
if newParticipantDirectory == 0
    return;
end


%% Expeirment nomenclature

% Pre-defined experiment mappings, adapt this to fit your respective
% experients, this is currently set up for the REF project and SEME
experimentMappings = containers.Map({'APS1', 'APS2', 'EPS1', 'EPS2', 'RS', 'FNC1', 'FNC2', 'FNC3', 'FPS1', 'FPS2', 'FPS3', 'FS1', 'FS2', 'FS3', 'FD1', 'FD2', 'FD3', 'FND1', 'FND2', 'FND3'}, ...
                                    {{'APS', 'run1'}, {'APS', 'run2'}, {'EPS', 'run1'}, {'EPS', 'run2'}, {'RS', 'run'}, ...
                                    {'FNC', 'run1'}, {'FNC', 'run2'}, {'FNC', 'run3'}, {'FPS', 'run1'}, {'FPS', 'run2'}, ...
                                    {'FPS', 'run3'}, {'FS', 'run1'}, {'FS', 'run2'}, {'FS', 'run3'}, ...
                                    {'FD', 'run1'}, {'FD', 'run2'}, {'FD', 'run3'}, ...
                                    {'FND', 'run1'}, {'FND', 'run2'}, {'FND', 'run3'}});

%% Loop to organise files.                                
                                
% Loop through all folders within the root directory
rootDirContents = dir(rootDirectory);
dirFlags = [rootDirContents.isdir];
subFolders = rootDirContents(dirFlags);
for i = 1 : length(subFolders)
    dirName = subFolders(i).name;
    if strcmp(dirName, '.') || strcmp(dirName, '..')
        continue;
    end
    dirPath = fullfile(rootDirectory, dirName);
    
    % Look for experiment subdirectories
    experimentDirs = dir(fullfile(dirPath, '*'));
    experimentDirFlags = [experimentDirs.isdir];
    experimentSubFolders = experimentDirs(experimentDirFlags);
    
    % Iterate through the experiment directories
    for j = 1 : length(experimentSubFolders)
        experimentDirName = experimentSubFolders(j).name;
        if strcmp(experimentDirName, '.') || strcmp(experimentDirName, '..')
            continue;
        end
        experimentDirPath = fullfile(dirPath, experimentDirName);
        
        % Find the JSON file
        jsonFiles = dir(fullfile(experimentDirPath, '*_*_description.json'));
        
        % for some reason there are hidden files named exactly the same
        % as our JSON file of interests (likely some MacOS nonsense)
        % We are going to filter these out
        
        % Filter out any hidden files starting with '._'
        jsonFiles = jsonFiles(~startsWith({jsonFiles.name}, '._'));
        
        % Debugging: Print out the found JSON files
        if length(jsonFiles) ~= 1
            fprintf('Debug: Found %d JSON files in %s\n', length(jsonFiles), experimentDirPath);
            for k = 1:length(jsonFiles)
                fprintf('Debug: JSON file found: %s\n', jsonFiles(k).name);
            end
        end
    
        % Ensure there is exactly one JSON file
        if length(jsonFiles) == 1
            jsonFilePath = fullfile(experimentDirPath, jsonFiles(1).name);
            
            % Read the JSON file
            data = jsondecode(fileread(jsonFilePath));
            
            % Extract subject name and experiment name from JSON data
            subjectName = data.subject;
            experimentName = data.experiment;
            
            % Get folder name and run name based on experimentMappings
            if isKey(experimentMappings, experimentName)
                mapping = experimentMappings(experimentName);
                folderName = mapping{1};
                runName = mapping{2};
            else
                folderName = 'UnknownExperiment';
                runName = '';
            end
            
            % Construct the final target path
            finalTargetPath = fullfile(newParticipantDirectory, subjectName, folderName);
            if ~isempty(runName)
                finalTargetPath = fullfile(finalTargetPath, runName);
            end
            
            % Check if the final target path already exists
            if exist(finalTargetPath, 'dir')
                fprintf('Skipping %s for %s: Already exists in target directory.\n', experimentName, subjectName);
                continue;
            end
            
            % Proceed with creating directories and copying files
            if ~exist(finalTargetPath, 'dir')
                mkdir(finalTargetPath);
            end
            
            % Copy the experiment directory to the new location
            copyfile(experimentDirPath, finalTargetPath);
            fprintf('Copied %s to %s\n', experimentDirPath, finalTargetPath);
        else
            fprintf('Skipping %s: Found %d JSON files.\n', experimentDirPath, length(jsonFiles));
        end
    end
end
end
