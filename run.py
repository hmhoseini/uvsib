from workflows.workflows import add_from_frontend, update_frontend

if __name__ == "__main__":
    dict_from_frontend_list = [
            {"user": "one@hzdr.de", "chemical_formula": "TiSe3", "process": "CO2RR", "metadata" : {"model": "MACE"}},
#            {"user": "two@hzdr.de", "chemical_formula": "TiSe2", "process":"WS", "metadata" : {"model": "MACE"}},
#            {"user": "three@hzdr.de", "chemical_formula": "TiSe8", "process": "CO2RR", "metadata" : {"model": "MACE"}},
#            {"user": "three@hzdr.de", "chemical_formula": "TiSe2", "metadata" : {"model": "MACE"}},
#            {"user": "three@hzdr.de", "chemical_formula": "NiSe2", "metadata" : {"model": "MACE"}},

             ]
    add_from_frontend(dict_from_frontend_list)
    update_frontend()
