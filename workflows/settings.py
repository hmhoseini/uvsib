import os
import yaml
from aiida.orm import Group
from aiida.manage.configuration import load_profile

load_profile()

this_directory = os.path.abspath(os.path.dirname(__file__))
uvsib_directory = os.path.split(this_directory)[0]

run_folder_group = Group.collection.get(label='uvsib_run_folder')
run_dir = run_folder_group.nodes[0].value

with open(os.path.join(run_dir, 'input.yaml'), 'r', encoding='utf8') as fhandle:
    inputs = yaml.safe_load(fhandle)

with open(os.path.join(run_dir, 'config.yaml'), 'r', encoding='utf8') as fhandle:
    configs = yaml.safe_load(fhandle)

api_key = configs['MP_API_KEY']['api_key']

code_folder_path =  os.path.join(uvsib_directory, 'codes')
mattergen_files_path = os.path.join(code_folder_path, 'mattergen', 'mattergen_files')
mattersim_files_path = os.path.join(code_folder_path, 'mattersim', 'mattersim_files')
mace_files_path = os.path.join(code_folder_path, 'mace', 'mace_files')
minimahopping_files_path = os.path.join(code_folder_path, 'minimahopping', 'mh_files')
vasp_files_path = os.path.join(code_folder_path, 'vasp', 'vasp_files')
