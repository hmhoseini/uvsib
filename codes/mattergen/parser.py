import tempfile
from ase.io import read
from aiida.common import exceptions
from aiida.parsers import Parser
from aiida.orm import Dict
from pymatgen.io.ase import AseAtomsAdaptor

class MatterGenParser(Parser):
    """
    Parser for MatterGenCalculation.

    Reads generated_crystals.extxyz directly from the retrieved folder
    into memory and converts all frames to StructureData.
    """

    def parse(self, **kwargs):
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        filename = 'generated_crystals.extxyz'
        if filename not in retrieved_folder.list_object_names():
            return self.exit_codes.ERROR_MISSING_OUTPUT

        ase_structures = []
        data = retrieved_folder.get_object_content('generated_crystals.extxyz')
        with tempfile.NamedTemporaryFile('w+', suffix='.extxyz', delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            ase_structures = read(tmp.name, index=':')
        if not ase_structures:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE

        adaptor = AseAtomsAdaptor()
        pmg_structures = []
        for ase_struct in ase_structures:
            pmg_struct = adaptor.get_structure(ase_struct)
            pmg_structures.append(pmg_struct.as_dict())

        output_structures = {"structures": pmg_structures}
        self.out("output_dict", Dict(dict=output_structures))
