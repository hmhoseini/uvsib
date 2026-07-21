"""ASE calculator factory used by relax / face_build / adsorbates / mh / nano_particles.

Shipped as a sidecar alongside each ``aiida.py`` script (see the CalcJob
``prepare_for_submission`` of mace / mattersim / upet / uma / nano_particles /
minimahopping). Each script does ``from _calculators import make_calculator``
and the workchain controls the backend + UMA task head via ``--ML_model`` and
``--task_name``.
"""

_UMA_TASKS = ("omat", "oc20", "oc22", "omol", "odac", "omc")
_MACE_TASKS = ("omat_pbe", "omol", "spice_wB97M", "rgd1_b3lyp", "oc20_usemppbe", "matpes_r2scan", "Default")


def make_calculator(ml_model, *, model=None, model_path=None, device="cuda",
                    task_name=None):
    """Build an ASE calculator from the workflow-supplied backend identifier.

    Parameters
    ----------
    ml_model : str
        Backend tag passed in via ``--ML_model``. Recognised tokens (substring
        match, matching the workchain dispatch):
        ``MACE``, ``uPET``, ``MatterSim``, ``UMA``.
    model : str | None
        HuggingFace-style model name (``--model``); used by uPET and UMA.
    model_path : str | None
        Local path to a checkpoint (``--model_path``); used by MACE and MatterSim.
    device : str
        Torch device string.
    task_name : str | None
        UMA task head. Defaults to ``omat`` when UMA is selected without an
        explicit task; ignored for the other backends. Must be one of
        ``{_UMA_TASKS}``.
    """
    if "MACE" in ml_model:
        if task_name not in _MACE_TASKS:
            raise ValueError(f"Unknown MACE task_name '{task_name}'. Expected one of: {', '.join(_MACE_TASKS)}.")
        from mace.calculators import mace_mp
        return mace_mp(model=model_path, default_dtype="float64", device=device, head=task_name)
        # from mace.calculators import MACECalculator
        # return MACECalculator(model_paths=model_path, device=device)

    if "uPET" in ml_model:
        from upet.calculator import UPETCalculator
        return UPETCalculator(model=model, device=device)

    if "MatterSim" in ml_model:
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(load_path=model_path, device=device)

    if "UMA" in ml_model:
        if task_name not in _UMA_TASKS:
            raise ValueError(f"Unknown UMA task_name '{task_name}'. Expected one of: {', '.join(_UMA_TASKS)}.")
        from fairchem.core import pretrained_mlip
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
        predictor = pretrained_mlip.get_predict_unit(model, device=device)
        return FAIRChemCalculator(predictor, task_name=task_name)

    raise ValueError(
        f"Unknown ML_model '{ml_model}'. Expected one of: "
        "MACE, uPET, UMA, MatterSim.")
