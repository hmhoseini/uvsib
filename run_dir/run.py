from uvsib.workflows.workflows import add_from_frontend, update_dbfrontend

if __name__ == "__main__":
    dict_from_frontend_list = [
#            {"user": "two@hzdr.de", "chemical_formula": "AgInS2", "reaction": "WS", "model": "MACE", "retry": True},
            {"user": "two@hzdr.de", "chemical_formula": "TiO", "reaction": "WS", "model": "MatterSim", "retry": False},
            {"user": "two@hzdr.de", "chemical_formula": "TiO2", "reaction": "WS", "model": "MatterSim", "retry": True},



             ]
    update_dbfrontend()
    add_from_frontend(dict_from_frontend_list)
