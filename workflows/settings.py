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

MAX_NUM_BULK = 5
MAX_NUM_SURF = 4
MAX_NUM_ADS = 2

EHULL_ML = 0.1
EHULL_SCAN = 0.1
DFT_FUNC = 'GGA'  # GGA/r2SCAN

# Run the GNoME (SAPS) generator in parallel with MatterGen in the gen + csp
# paths. Opt-in via input.yaml (`gnome: {enabled: true}`); defaults off so
# existing runs without the block are unaffected.
GNOME_PARALLEL = bool(inputs.get('gnome', {}).get('enabled', False))

# Run the MatterGen generator at all (de-novo gen + CSP). Toggles
# MatterGen_generate and MatterGen_CSP together. Configured in input.yaml under
# `mattergen:`; defaults on so existing runs without the block are unaffected.
# With this off and `gnome` on, generation runs GNoME-only; at least one
# generator must be enabled.
MATTERGEN_ENABLED = bool(inputs.get('mattergen', {}).get('enabled', True))

# Classify all generated structures by synthesizability (thermo + reaction +
# PU) as a MainWorkChain stage after the phase diagram. Configured in input.yaml
# under `synthesizability:`; enabled by default (post-processing only, no jobs).
SYNTH_ENABLED = bool(inputs.get('synthesizability', {}).get('enabled', False))

# Run adaptive kinetic Monte Carlo after adsorbate screening. This is opt-in
# because dimer searches are much more expensive than the CHE adsorbate pass.
AKMC_ENABLED = bool(inputs.get('akmc', {}).get('enabled', True))

# Soft stop: gracefully end the MainWorkChain after the generation/phase-diagram/
# synthesizability stages, before the surface builder (and adsorbates) start.
# Opt-in via input.yaml (`soft_stop: {before_surface_builder: true}`); absent or
# false -> the full pipeline runs as before.
SOFT_STOP_BEFORE_SURFACE = bool(inputs.get('soft_stop', {}).get('before_surface_builder', False))

_PD_VERIFICATION = False
_SKIP_CSP = False
_SKIP_GEN = False

code_folder_path =  os.path.join(uvsib_directory, 'codes')
files_path = os.path.join(code_folder_path, 'files')
molecular_reference_files = os.path.join(code_folder_path, 'files', 'molecular_references')
