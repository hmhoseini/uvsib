from aiida.orm import Str, List
from aiida_submission_controller import BaseSubmissionController
from workchains.main import MainWorkChain


class MainSubmissionController(BaseSubmissionController):
    """ SubmissionController
    """
    def __init__(self,
            uuid,
            chemical_formula,
            chemical_systems,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.uuid = uuid
        self.chemical_formula = chemical_formula
        self.chemical_systems = chemical_systems
    def get_extra_unique_keys(self):
        """ Return a tuple of the keys of the unique extras that
            will be used to uniquely identify your workchains
        """
        return ('uuid', 'chemical_formula')

    def get_all_extras_to_submit(self):
        """ Return a *set* of the values of all extras uniquely
            identifying all simulations that you want to submit.
            Each entry of the set must be a tuple, in same order
            as the keys returned by get_extra_unique_keys().
            Note: for each item, pass extra values as tuples
        """
        uuid = self.uuid
        chemical_formula = self.chemical_formula
        all_extras = set()
        all_extras.add((chemical_formula, uuid))
        return all_extras

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """ Return the inputs and the process class for the process
            to run, associated a given tuple of extras values.
            Param: extras_values: a tuple of values of the extras,
            in same order as the keys returned by get_extra_unique_keys().
        """
        inputs = {"chemical_formula": Str(extras_values[0]),
                  "chemical_systems": List(list=self.chemical_systems),
                  "uuid": Str(extras_values[1])}
        return inputs, MainWorkChain
