import os

root_dir = 'renametothepath' # change this to the path of the specific folder

for subdir, dirs, files in os.walk(root_dir):
    subdir_list = subdir.split('/')[len(root_dir.split('/')):]
    new_subdir_name = '_'.join(subdir_list)
    for file in files:
        if file.endswith('.mp4'):
            file_path = os.path.join(subdir, file)
            new_file_name = f'{new_subdir_name}_{file}'
            new_file_path = os.path.join(subdir, new_file_name)
            os.rename(file_path, new_file_path)
        else:
            file_path = os.path.join(subdir, file)
            os.remove(file_path)
