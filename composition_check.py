from itertools import product
from pymatgen.core.composition import Composition

oxidation_states_dict = {
 'H': [-1, 1],
 'Li': [-1, 1],
 'Be': [1, 2],
 'B': [-5, -1, 1, 2, 3],
 'C': [-4, -3, -2, -1, 1, 2, 3, 4],
 'N': [-3, -2, -1, 1, 2, 3, 4, 5],
 'O': [-2, -1, 1, 2],
 'F': [-1],
 'Na': [-1, 1],
 'Mg': [1, 2],
 'Al': [-2, -1, 1, 2, 3],
 'Si': [-4, -3, -2, -1, 1, 2, 3, 4],
 'P': [-3, -2, -1, 1, 2, 3, 4, 5],
 'S': [-2, -1, 1, 2, 3, 4, 5, 6],
 'Cl': [-1, 1, 2, 3, 4, 5, 6, 7],
 'K': [-1, 1],
 'Ca': [1, 2],
 'Sc': [1, 2, 3],
 'Ti': [-2, -1, 1, 2, 3, 4],
 'V': [-3, -1, 1, 2, 3, 4, 5],
 'Cr': [-4, -2, -1, 1, 2, 3, 4, 5, 6],
 'Mn': [-3, -2, -1, 1, 2, 3, 4, 5, 6, 7],
 'Fe': [-4, -2, -1, 1, 2, 3, 4, 5, 6, 7],
 'Co': [-3, -1, 1, 2, 3, 4, 5],
 'Ni': [-2, -1, 1, 2, 3, 4],
 'Cu': [-2, 1, 2, 3, 4],
 'Zn': [-2, 1, 2],
 'Ga': [-5, -4, -3, -2, -1, 1, 2, 3],
 'Ge': [-4, -3, -2, -1, 1, 2, 3, 4],
 'As': [-3, -2, -1, 1, 2, 3, 4, 5],
 'Se': [-2, -1, 1, 2, 3, 4, 5, 6],
 'Br': [-1, 1, 2, 3, 4, 5, 7],
 'Rb': [-1, 1],
 'Sr': [1, 2],
 'Y': [1, 2, 3],
 'Zr': [-2, 1, 2, 3, 4],
 'Nb': [-3, -1, 1, 2, 3, 4, 5],
 'Mo': [-4, -2, -1, 1, 2, 3, 4, 5, 6],
 'Tc': [-1, 1, 2, 3, 4, 5, 6, 7],
 'Ru': [-2, 1, 2, 3, 4, 5, 6, 7, 8],
 'Rh': [-3, -1, 1, 2, 3, 4, 5, 6, 7],
 'Pd': [1, 2, 3, 4, 5],
 'Ag': [-2, -1, 1, 2, 3],
 'Cd': [-2, 1, 2],
 'In': [-5, -2, -1, 1, 2, 3],
 'Sn': [-4, -3, -3, -2, 1, 2, 3, 4],
 'Sb': [-3, -2, -1, 1, 2, 3, 4, 5],
 'Te': [-2, -1, 1, 2, 3, 4, 5, 6],
 'I': [-1, 1, 2, 3, 4, 5, 6, 7],
 'Cs': [-1, 1],
 'Ba': [1, 2],
 'Hf': [-2, 1, 2, 3, 4],
 'Ta': [-3, -1, 1, 2, 3, 4, 5],
 'W': [-4, -2, -1, 1, 2, 3, 4, 5, 6],
 'Re': [-3, -1, 1, 2, 3, 4, 5, 6, 7],
 'Os': [-4, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8],
 'Ir': [-3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 'Pt': [-3, -2, -1, 1, 2, 3, 4, 5, 6],
 'Au': [-3, -2, -1, 1, 2, 3, 5],
 'Hg': [-2, 1, 2],
 'Tl': [-5, -2, -1, 1, 2, 3],
 'Pb': [-4, -2, -1, 1, 2, 3, 4],
 'Bi': [-3, -2, -1, 1, 2, 3, 4, 5],
 'Po': [-2, 2, 4, 5, 6],
 'At': [-1, 1, 3, 5, 7]
}

def is_charge_neutral(comp: Composition):
    """
    Check whether a composition can be charge neutral
    using common oxidation states.

    Parameters
    ----------
    comp : pymatgen Composition

    """

    if comp.is_element:
        return True,  "Charge neutral"

    elements = list(comp.elements)
    amounts = [comp[el] for el in elements]

    oxidation_state_lists = []
    for el in elements:
        try:
            ox_states = oxidation_states_dict.get(el.symbol)
            if not ox_states:
                return False, f"Oxidation state of {el} is not known."
        except:
            return False, f"Oxidation state of {el} is not known."
        oxidation_state_lists.append(ox_states)
    for ox_state_combo in product(*oxidation_state_lists):
        total_charge = sum(ox * amt for ox, amt in zip(ox_state_combo, amounts))
        if total_charge == 0:
            return True, "Charge neutral"
    return False, "Input composition is not charge neutral"

def validate_composition(comp_string: str, max_elements: int):
    """
    Validate whether a string is a valid pymatgen composition,
    whether the number of unique elements is below a limit,
    and whether the composition is charge neutral.

    Parameters
    ----------
    comp_string : str
        Composition string (e.g. "Fe2O3")
    max_elements : int
        Maximum allowed number of unique elements

    Returns
    -------
    tuple
        (True, message) if valid
        (False, error_message) if invalid
    """

    # Check input type
    if not isinstance(comp_string, str):
        return False, "Input composition must be a string."

    try:
        # Parse composition
        comp = Composition(comp_string)

    except Exception as e:
        return False, f"Invalid composition: {e}"

    # Count unique elements
    num_elements = len(comp.elements)

    if num_elements > max_elements:
        return (
            False,
            f"Composition contains {num_elements} elements, "
            f"which exceeds the maximum allowed ({max_elements})."
        )

    # Check charge neutrality
    neutral, neutral_msg = is_charge_neutral(comp)

    if not neutral:
        return (False, neutral_msg)

    return (
        True, "Valid and charge-neutral composition")


# Example usage
if __name__ == "__main__":

    tests = [
        ("Fe2O3", 3),
        ("NaCl", 2),
        ("LiFePO4", 4),
        ("FeO9", 3),      # likely non-neutral
        ("Invalid123", 3),
        ("NaCl2", 2),
    ]

    for formula, max_el in tests:
        valid, message = validate_composition(formula, max_el)
        print(f"{formula} -> {valid}: {message}")
