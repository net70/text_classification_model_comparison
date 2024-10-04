from .env import base_path
import json
import re
import os
import sys

def find_file(file_name: str, root_directory: str = base_path):
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if file_name in filenames:
            return os.path.join(dirpath, file_name)
    return None