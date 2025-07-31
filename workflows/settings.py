import os
import yaml
from aiida.orm import Group
from aiida.manage.configuration import load_profile

load_profile()

this_directory = os.path.abspath(os.path.dirname(__file__))
uvsib_directory = os.path.split(this_directory)[0]

with open(os.path.join(this_directory,'input.yaml'), 'r', encoding='utf8') as fhandle:
    inputs = yaml.safe_load(fhandle)

with open(os.path.join(this_directory, 'config.yaml'), 'r', encoding='utf8') as fhandle:
    configs = yaml.safe_load(fhandle)

api_key = configs['MP_API_KEY']['api_key']

with open(os.path.join(this_directory, 'groups.yaml'), 'r', encoding='utf8') as fhandle:
    groups = yaml.safe_load(fhandle)

code_folder_path =  os.path.join(uvsib_directory, 'codes')
mattergen_files_path = os.path.join(code_folder_path, 'mattergen_files')
mattersim_files_path = os.path.join(code_folder_path, 'mattersim', 'mattersim_files')
mace_files_path = os.path.join(code_folder_path, 'mace', 'mace_files')

