from aiida.orm import Str
from aiida_submission_controller import BaseSubmissionController
from codes.mattergen.workchain import MatterGenWorkChain


class MatterGenSubmissionController(BaseSubmissionController):
    """ SubmissionController
    """
    def __init__(self,
            chemical_system,
            uuid,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.chemical_system = chemical_system
        self.uuid = uuid

    def get_extra_unique_keys(self):
        """ Return a tuple of the keys of the unique extras that
            will be used to uniquely identify your workchains
        """
        return ('chemical_system', 'uuid')

    def get_all_extras_to_submit(self):
        """ Return a *set* of the values of all extras uniquely
            identifying all simulations that you want to submit.
            Each entry of the set must be a tuple, in same order
            as the keys returned by get_extra_unique_keys().
            Note: for each item, pass extra values as tuples
        """
        chemical_system = self.chemical_system
        uuid = self.uuid
        all_extras = set()
        all_extras.add((chemical_system, uuid))
        return all_extras

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """ Return the inputs and the process class for the process
            to run, associated a given tuple of extras values.
            Param: extras_values: a tuple of values of the extras,
            in same order as the keys returned by get_extra_unique_keys().
        """
        inputs = {"chemical_system": Str(extras_values[0])}
        return inputs, MatterGenWorkChain
