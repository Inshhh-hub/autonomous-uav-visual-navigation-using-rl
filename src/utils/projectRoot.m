function root = projectRoot()
%PROJECTROOT  Absolute path of the repository root folder.
root = string(fileparts(fileparts(fileparts(mfilename("fullpath")))));
end
