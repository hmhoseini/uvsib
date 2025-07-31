from workchains.launch_calculations import MainSubmissionController

def submit_mainworkchain(chemical_formula, chemical_systems, uuid):
    controller = MainSubmissionController(
        group_label='wf_test',
        max_concurrent=5,
        chemical_formula = chemical_formula,
        chemical_systems = chemical_systems,
        uuid = uuid
    )
    controller.submit_new_batch(dry_run=False)
    return controller.get_all_submitted_pks()
