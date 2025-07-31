import os
from pymatgen.core import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from aiida.orm import Int, Str, List, Dict, Code
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory
from aiida_pythonjob import PythonJob
from db.tables import DBChemsys, DBStructure, DBStructureVersion
from db.query import query_by_columns
from db.session import get_session
from db.utils import update_row, delete_row, get_chemical_systems, add_structures
from codes.utils import select_charge_neutral, get_structures_from_mpdb
from workchains.pythonjob_inputs import get_pythonjob_input
from workflows import settings

class DFTVerificationWorkChain(WorkChain):
    """Work chain for verification"""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("uuid_list", valid_type=List)

        spec.outline(
            cls.setup,
            cls.run_pbe,
            cls.pbe_results,
            cls.run_scan,
            cls.scan_results,
            cls.run_hse,
            cls.hse_resutls,
        )
        spec.exit_code(300,
            "ERROR_CALCULATION_FAILED",
            message="The calculation did not finish successfully"
        )

    def setup(self):
        self.ctx.uuid_list = self.inputs.uuid_list.get_list()

    def run_pbe(self):
        """ """
        vasppbeworkflow = VASPPBERelaxWorkchain

        for a_uuid in self.ctx.uuid_list:
            result = (session.query(DBStructureVersion)
                      .join(DBStructure)
                      .filter(DBStructure.uuid == a_uuida)
                      .filter(DBStructureVersion.method == "MatterSim")
                      .first()
                      )
            builder = vasppbeworkflow.get_builder()
            builder.structure = result.structure
            builder.protocol = pbe_protocol
            future = self.submit(builder)
            self.to_context(**{f"a_uuid": future})

