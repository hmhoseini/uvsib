import os
import subprocess
from custodian import Custodian
from custodian.custodian import Job
from custodian.vasp.handlers import VaspErrorHandler, MeshSymmetryErrorHandler, UnconvergedErrorHandler, \
    NonConvergingErrorHandler, PotimErrorHandler, PositiveEnergyErrorHandler, FrozenJobErrorHandler, StdErrHandler
from custodian.vasp.validators import VasprunXMLValidator, VaspFilesValidator


"""
Run VASP custodian, fixes most runtime errors, uses default handlers and validators
"""


class VaspJob(Job):
    def __init__(self, vasp_cmd):
        self.run_dir = os.getcwd()
        self.vasp_cmd = vasp_cmd
        self.std_out = 'vasp_output'  # not compatible to handlers
        self.std_err = 'vasp_output'  # not compatible to handlers, errors go into vasp_output because of the
                                      # aiida scheduler implementation, in custodian.handlers now changed to vasp_output

    def setup(self, directory="./"):
        pass

    def run(self, directory="./"):
        with open(self.std_out, 'w') as sout, open(self.std_err, 'w', buffering=1) as serr:
            p = subprocess.Popen(self.vasp_cmd.split(), stdout=sout, stderr=serr)
        return p

    def postprocess(self, directory="./"):
        pass


handlers = [VaspErrorHandler(), MeshSymmetryErrorHandler(), UnconvergedErrorHandler(),
            NonConvergingErrorHandler(), PotimErrorHandler(),
            PositiveEnergyErrorHandler(), FrozenJobErrorHandler(), StdErrHandler()]
validators = [VasprunXMLValidator(), VaspFilesValidator()]

vasp_cmd = 'srun vasp_std'
c = Custodian(handlers, [VaspJob(vasp_cmd=vasp_cmd)], validators=validators, max_errors=5, gzipped_output=False)
c.run()
