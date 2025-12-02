import tempfile
import numpy as np
from pymatgen.io.vasp import Vasprun
from matminer.featurizers.bandstructure import BranchPointEnergy

def load_vasprun_from_content(wch):
    vasprun_str = wch.called[-1].outputs.retrieved.get_object_content("vasprun.xml")
    with tempfile.NamedTemporaryFile(mode="w", delete=True) as tmp:
        tmp.write(vasprun_str)
        tmp.flush()
        vasprun = Vasprun(tmp.name)
    return vasprun_str, vasprun

def branch_point_energy_window(bs, window_vb, window_cb):
    """Calculate BPE using all bands within window above CBM and below VBM"""
    if bs.is_metal():
        return None

    cbm = bs.get_cbm()["energy"]
    vbm = bs.get_vbm()["energy"]

    total_bpe = 0.0
    counted_kpoints = 0

    for spin in bs.bands.keys():
        for i in range(len(bs.kpoints)):
            energies_at_k = bs.bands[spin][:, i]

            valence_states = energies_at_k[(energies_at_k <= vbm) & (energies_at_k >= vbm - window_vb)]
            conduction_states = energies_at_k[(energies_at_k >= cbm) & (energies_at_k <= cbm + window_cb)]

            if len(valence_states) == 0 or len(conduction_states) == 0:
                continue

            e_v_k = np.mean(valence_states)
            e_c_k = np.mean(conduction_states)
            total_bpe += 0.5 * (e_v_k + e_c_k)
            counted_kpoints += 1

    if counted_kpoints == 0:
        return None

    return total_bpe / counted_kpoints

def branch_point_energy_dos_centroid(vasprun, bs, window_vb, window_cb):
    """Calculate BPE from DOS centroids near VBM and CBM"""
    dos = vasprun.complete_dos
    vbm = bs.get_vbm()['energy']
    cbm = bs.get_cbm()['energy']

    e = np.array(dos.energies)
    d = np.array(dos.densities[list(dos.densities.keys())[0]])
    #d = np.array(dos.densities['1'])  # total DOS (spin-summed if spin-unpolarized)

    # Valence centroid
    mask_v = (e >= vbm - window_vb) & (e <= vbm)
    if d[mask_v].sum() == 0:
        return None
    Ev = (e[mask_v] * d[mask_v]).sum() / d[mask_v].sum()

    # Conduction centroid
    mask_c = (e >= cbm) & (e <= cbm + window_cb)
    if d[mask_c].sum() == 0:
        return None
    Ec = (e[mask_c] * d[mask_c]).sum() / d[mask_c].sum()

    return 0.5 * (Ev + Ec)

def count_bands_within_window(bs, window_vb, window_cb):
    """
    Count number of conduction bands within +window_ev above CBM
    and number of valence bands within -window_ev below VBM
    """
    cbm_energy = bs.get_cbm()["energy"]
    vbm_energy = bs.get_vbm()["energy"]

    n_valence = 0
    n_conduction = 0

    for spin in bs.bands:
        for kpoint_energies in bs.bands[spin]:
            avg_energy = sum(kpoint_energies) / len(kpoint_energies)

            if avg_energy <= vbm_energy and (vbm_energy - avg_energy) <= window_vb:
                n_valence += 1
            elif avg_energy >= cbm_energy and (avg_energy - cbm_energy) <= window_cb:
                n_conduction += 1

    return n_valence, n_conduction

def get_band_info(wch, w_v=3, w_c=3):
    vr_str, vr = load_vasprun_from_content(wch)
    efermi = wch.outputs.misc.dict.fermi_level
    if efermi:
        bs = vr.get_band_structure(efermi=efermi)
    else:
        return vr_str, {}

    if bs.is_metal():
        return vr_str, {}

    vbm_info = bs.get_vbm()
    cbm_info = bs.get_cbm()

    bpe_band = branch_point_energy_window(bs, w_v, w_c)

    bpe_dos = branch_point_energy_dos_centroid(vr, bs, w_v, w_c)

    n_v, n_c = count_bands_within_window(bs, w_v, w_c)
    n_valence = max(1, n_v)
    n_conduction = max(1, n_c)
    try:
        featurizer = BranchPointEnergy(n_valence, n_conduction)
        bpe_default = round(featurizer.featurize(bs)[0], 2)
    except:
        bpe_default = None

    band_info_dict = bs.get_band_gap()
    band_info_dict["energy"]  = round(band_info_dict["energy"], 2)
    band_info_dict.update({
        "vbm": vbm_info["energy"],
        "cbm": cbm_info["energy"],
        "bpe_window": round(bpe_band, 2),
        "bpe_dos": round(bpe_dos, 2),
        "bpe_matminer": bpe_default}
    )
    return vr_str, band_info_dict
