def is_data_available(chemical_systems, timeout = 36000):
    """Check if data for a phase diagram is available.
    If data is available, return True, otherwise sleep. Timeout 10 h.
    """
    import time
#    from aiida import load_profile
    from uvsib.db.tables import DBChemsys
    from uvsib.db.utils import query_by_columns

#    load_profile()
    tstart = time.time()
    while True:
        missing_data = []
        for chemical_system in chemical_systems:
            try:
                result = query_by_columns(DBChemsys,
                                          {"chemsys": chemical_system,
                                           "gen_structures": "Ready"
                                           }
                )
                if not result:
                    missing_data.append(chemical_system)
            except:
                return {"moveon": False}

        if not missing_data:
            return {"moveon": True}
        if time.time() - tstart > timeout:
            return {"moveon": False}
        time.sleep(600)

def wait_sleep(timeout = 3600):
    """Sleep"""
    import time
    tstart = time.time()
    while True:
        if time.time() - tstart > timeout:
            return
        time.sleep(600)
