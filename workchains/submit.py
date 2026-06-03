import uuid
from uvsib.workchains.launch_calculations import MainSubmissionController


def submit_mainworkchain(chemical_formula, chemical_systems, reaction, reaction_path,
                         nano=False, similarities=None, sqs=False):
    controller = MainSubmissionController(
        group_label='wf_test',
        max_concurrent=10,
        uuid_str=str(uuid.uuid4()),
        chemical_formula=chemical_formula,
        chemical_systems=chemical_systems,
        reaction=reaction,
        reaction_path=reaction_path,
        nanoparticles=nano,
        similarities=similarities,
        sqs=sqs
    )
    controller.submit_new_batch(dry_run=False)
    return controller.get_all_submitted_pks()
