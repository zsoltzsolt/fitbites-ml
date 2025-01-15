import shutil
import os

def save_temp_file(file, temp_file_path):
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

def remove_temp_file(temp_file_path):
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
