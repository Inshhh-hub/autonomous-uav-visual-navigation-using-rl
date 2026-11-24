function setupProject()
%SETUPPROJECT  Add the project folders to the MATLAB path and check requirements.
%   Run once per MATLAB session from the project root:  >> setupProject

root = fileparts(mfilename("fullpath"));
addpath(root);
addpath(fullfile(root, "config"));
addpath(genpath(fullfile(root, "src")));
addpath(fullfile(root, "scripts"));
for f = ["data" "checkpoints" "results"]
    if ~isfolder(fullfile(root, f))
        mkdir(fullfile(root, f));
    end
end

if isMATLABReleaseOlderThan("R2024a")
    warning("uavnav:release", "This project targets MATLAB R2024a or newer (you are on %s).", version("-release"));
end
v = ver;
installed = string({v.Name});
required = ["Reinforcement Learning Toolbox" "Deep Learning Toolbox"];
optional = ["UAV Toolbox" "Parallel Computing Toolbox"];
for p = required
    if ~any(installed == p)
        warning("uavnav:toolbox", "%s is required but was not found.", p);
    end
end
for p = optional
    if ~any(installed == p)
        fprintf("  (optional) %s not found - related features are disabled.\n", p);
    end
end
fprintf("UAV visual-navigation project ready. Try:  demoSimulator   or   runAll\n");
end
