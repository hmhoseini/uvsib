# OER: Key "reaction": {"OER": None}
triggers OER computation and stores overpotential in the psql


# NOx electroreduction reaction pathways on metal surfaces. Key: "reaction": {"NOXRR": "no3_to_n2"}
Provides a library of NOx reduction intermediates and reaction pathways
(CHE model) that can be placed on *any* ASE Atoms surface slab.

NOx covered: NO, NO₂, NO₃⁻ as starting species.

Pathways implemented
--------------------
- ``'no_dissociative'``  : *NO → *N + *O → N₂(g)              (Ru, Rh, Ir)
- ``'no_to_nh3_noh'``   : *NO → *NOH → *N → *NH₂ → NH₃       (Cu, Fe)
- ``'no_to_nh3_nhoh'``  : *NO → *NOH → *NHOH → *NH₂OH → NH₃  (Cu, hydroxylamine route)
- ``'no_to_n2o'``        : 2*NO → *ONNO → N₂O + *O            (Pt, Pd automotive)
- ``'no2_to_no'``        : *NO₂ → *NO + *O                     (prereduction step)
- ``'no3_to_nh3'``       : *NO₃ → *NO₂ → *NO → … → NH₃        (eNO3RR, Cu)
- ``'no3_to_n2'``        : *NO₃ → *NO₂ → *NO → *N → N₂        (eNO3RR, Ru)

References
----------
Gao et al. *Nat. Chem.* **9**, 547 (2017).
Liu et al. *Nat. Commun.* **12**, 5797 (2021).
Wang et al. *J. Am. Chem. Soc.* **142**, 5702 (2020).
van 't Veer et al. *J. Phys. Chem. C* **124**, 22 (2020).
Pérez-Ramírez & López *Nat. Catal.* **2**, 971 (2019).


# CO2 electroreduction reaction pathways on metal surfaces. "reaction": {"CO2RR": "co2_to_ch3oh"}
Provides a literature-grounded library of CO2RR intermediates and reaction
pathways (CHE model, Nørskov group and follow-up work) that can be placed on
*any* ASE Atoms surface slab.

Pathways implemented
--------------------
- ``'co2_to_co'``    : CO2 → *COOH → *CO → CO(g)            (Au, Ag, Zn)
- ``'co2_to_hcooh'`` : CO2 → *OCHO → HCOOH(aq)              (formate, Pd, In)
- ``'co_to_ch4'``    : *CO → *CHO → *CHOH → *CH2 → *CH3 → CH4(g)  (Cu)
- ``'co_to_ch3oh'``  : *CO → *CHO → *CHOH → *CH2OH → CH3OH(g)     (Cu)
- ``'co2_to_ch4'``   : CO2 → CH4 full pathway on Cu
- ``'co2_to_ch3oh'`` : CO2 → CH3OH full pathway on Cu
- ``'co2_to_c2h4'``  : 2 *CO → *OCCO → … → C2H4(g)          (Cu C–C coupling)

Each intermediate is an ASE ``Atoms`` object (binding atom at index 0) that
can be placed on a slab via :func:`place_intermediate`.

References
----------
Peterson et al. *Energy Environ. Sci.* **3**, 1311 (2010).
Kuhl et al. *J. Am. Chem. Soc.* **136**, 14107 (2014).
Montoya et al. *ChemSusChem* **8**, 2180 (2015).
Goodpaster et al. *J. Phys. Chem. Lett.* **7**, 1471 (2016).


# Nano particle: 'nano_particles': False / True
the key has to be present or the frontend update will fail. 
In case of True will bypass the catalysis workchain completely at this 
point and produce nano-particles. More documentation for functionality 
will follow.
