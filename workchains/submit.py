import uuid
from uvsib.workchains.launch_calculations import MainSubmissionController

def submit_mainworkchain(chemical_formula, chemical_systems, model, reaction):
    controller = MainSubmissionController(
        group_label='wf_test',
        max_concurrent=5,
        uuid_str = str(uuid.uuid4()),
        chemical_formula = chemical_formula,
        chemical_systems = chemical_systems,
        model = model,
        reaction = reaction
    )
    controller.submit_new_batch(dry_run=False)
    return controller.get_all_submitted_pks()
