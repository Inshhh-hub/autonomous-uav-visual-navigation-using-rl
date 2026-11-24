function ensureFolder(filePathOrFolder)
%ENSUREFOLDER  Create the folder of a file path (or the folder itself) if missing.
p = string(filePathOrFolder);
[folder, ~, ext] = fileparts(p);
if strlength(ext) == 0
    folder = p;
end
if strlength(folder) > 0 && ~isfolder(folder)
    mkdir(folder);
end
end
