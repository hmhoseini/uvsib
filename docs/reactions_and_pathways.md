# Reactions and pathways

This document is the registry of (photo)electrocatalytic reactions and reaction
pathways implemented in `codes/files/adsorbates.py`, plus a list of natural
extensions that fit the same template. All implemented reactions follow the
computational hydrogen electrode (CHE) framework (Nørskov et al. 2004).

The dispatch entry point is `generate_adsorbed_structures(reaction, pathway_name)`
in `codes/files/adsorbates.py`.

## Molecular reference structures

Gas-phase / solid-state reference cells live in
`codes/files/molecular_references/` as VASP POSCARs and are loaded once at
import time. Provenance:

| Ref     | Source                              | Notes                              |
|---------|-------------------------------------|------------------------------------|
| h2      | Materials Project mp-* (P2_12_12_1) | Z=4                                |
| h2o     | Materials Project mp-* (Cmc2_1)     | Z=8 (ice-like)                     |
| n2      | Materials Project mp-* (P2_13)      | Z=4 (alpha-N2)                     |
| nh3     | Materials Project mp-* (P2_13)      | Z=4 (cubic ammonia)                |
| n2o     | Materials Project mp-* (Pnma)       | Z=4                                |
| co2     | Materials Project mp-* (Pa-3)       | Z=4 (dry ice)                      |
| ch4     | Materials Project mp-* (I-43m)      | Z=2                                |
| ch3oh   | hand-built gas-phase, 15 Å vacuum cube | equilibrium geometry, single molecule |
| cl2     | Materials Project mp-22848 (Cmce)   | Z=2, stable                        |
| o2      | Materials Project mp-723285 (P4_12_12) | Z=8, near-stable                |
| h2o2    | Materials Project mp-28015 (P4_12_12)  | Z=16                            |

When a new gas-phase reference is needed, prefer the Materials Project crystal
(matches the project's per-formula-unit-energy convention) and use a vacuum-cube
single molecule only when no MP entry is suitable.

---

# Currently implemented reactions

## OER — oxygen evolution
Dispatch key: `{"OER": None}` (pathway name not used).

Triggers OER computation and stores the overpotential in PostgreSQL. Single
4e⁻ associative pathway: `* → *OH → *O → *OOH → *` releasing O₂.

References
----------
Nørskov et al. *J. Phys. Chem. B* **108**, 17886 (2004).
Man et al. *ChemCatChem* **3**, 1159 (2011) — universal scaling on perovskites.
Rossmeisl et al. *J. Electroanal. Chem.* **607**, 83 (2007).

## CO2RR — CO₂ electroreduction
Dispatch key: `{"CO2RR": "<pathway_name>"}`.

Pathways implemented
--------------------
- ``'co2_to_co'``    : CO₂ → \*COOH → \*CO → CO(g)                    (Au, Ag, Zn)
- ``'co2_to_hcooh'`` : CO₂ → \*OCHO → HCOOH(aq)                       (formate; Pd, In, Sn)
- ``'co_to_ch4'``    : \*CO → \*CHO → \*CHOH → \*CH → \*CH₂ → \*CH₃ → CH₄(g)  (Cu)
- ``'co_to_ch3oh'``  : \*CO → \*CHO → \*CHOH → \*CH₂OH → CH₃OH(g)              (Cu, Mo)
- ``'co2_to_ch4'``   : full CO₂ → CH₄ chain on Cu
- ``'co2_to_ch3oh'`` : full CO₂ → CH₃OH chain on Cu/Mo
- ``'co2_to_c2h4'``  : 2 \*CO → \*OCCO → … → C₂H₄(g)                  (Cu, C–C coupling)

References
----------
Peterson et al. *Energy Environ. Sci.* **3**, 1311 (2010).
Kuhl et al. *J. Am. Chem. Soc.* **136**, 14107 (2014).
Montoya et al. *ChemSusChem* **8**, 2180 (2015).
Goodpaster et al. *J. Phys. Chem. Lett.* **7**, 1471 (2016).

## NOXRR — NOx electroreduction
Dispatch key: `{"NOXRR": "<pathway_name>"}`. NOx covered: NO, NO₂, NO₃⁻.

Pathways implemented
--------------------
- ``'no_dissociative'`` : \*NO → \*N + \*O → N₂(g)               (Ru, Rh, Ir)
- ``'no_to_nh3_noh'``   : \*NO → \*NOH → \*N → \*NH₂ → NH₃        (Cu, Fe)
- ``'no_to_nh3_nhoh'``  : \*NO → \*NOH → \*NHOH → \*NH₂OH → NH₃   (Cu, hydroxylamine route)
- ``'no_to_n2o'``       : 2 \*NO → \*ONNO → N₂O + \*O             (Pt, Pd automotive)
- ``'no2_to_no'``       : \*NO₂ → \*NO + \*O                      (prereduction step)
- ``'no3_to_nh3'``      : \*NO₃ → \*NO₂ → \*NO → … → NH₃          (eNO3RR, Cu)
- ``'no3_to_n2'``       : \*NO₃ → \*NO₂ → \*NO → \*N → N₂         (eNO3RR, Ru)

References
----------
Gao et al. *Nat. Chem.* **9**, 547 (2017).
Liu et al. *Nat. Commun.* **12**, 5797 (2021).
Wang et al. *J. Am. Chem. Soc.* **142**, 5702 (2020).
van 't Veer et al. *J. Phys. Chem. C* **124**, 22 (2020).
Pérez-Ramírez & López *Nat. Catal.* **2**, 971 (2019).

## CER — chlorine evolution
Dispatch key: `{"CER": "<pathway_name>"}`. CHE referenced to the Cl⁻/Cl₂ couple
(E° = 1.36 V vs SHE in 1 M HCl). Electrochemical steps consume Cl⁻ instead of H⁺
(`protons=0, electrons=-1`).

Pathways implemented
--------------------
- ``'volmer_tafel'``     : \* + Cl⁻ → \*Cl, then 2 \*Cl → Cl₂(g) + 2 \*    (Pt, Ru, Pd)
- ``'volmer_heyrovsky'`` : \* + Cl⁻ → \*Cl, then \*Cl + Cl⁻ → Cl₂(g) + \*  (RuO₂, IrO₂)
- ``'krishtalik'``       : \*O + Cl⁻ → \*OCl, then \*OCl + Cl⁻ → Cl₂ + \*O  (RuO₂, IrO₂, Co₃O₄)

References
----------
Hansen et al. *Phys. Chem. Chem. Phys.* **12**, 283 (2010).
Exner et al. *Electrocatalysis* **6**, 163 (2015).
Exner et al. *ChemElectroChem* **3**, 1607 (2016).
Karlsson & Cornell *Chem. Rev.* **116**, 2982 (2016).
Lim et al. *Nat. Commun.* **11**, 412 (2020).
Kim et al. *Angew. Chem. Int. Ed.* **64**, e202417293 (2025).

## HER — hydrogen evolution
Dispatch key: `{"HER": "<pathway_name>"}`. Foundational reaction; \*H is the
single intermediate, |ΔG_H\*| ≈ 0 marks the volcano apex.

Pathways implemented
--------------------
- ``'volmer_tafel'``     : \* + H⁺ + e⁻ → \*H, then 2 \*H → H₂(g) + 2 \*    (Pt, Pd, Ir)
- ``'volmer_heyrovsky'`` : \* + H⁺ + e⁻ → \*H, then \*H + H⁺ + e⁻ → H₂(g) + \*  (MoS₂, Au, Ag, Ni–Mo)

References
----------
Nørskov et al. *J. Electrochem. Soc.* **152**, J23 (2005).
Greeley et al. *Nat. Mater.* **5**, 909 (2006).
Conway & Tilak *Electrochim. Acta* **47**, 3571 (2002).
Skúlason et al. *J. Phys. Chem. C* **114**, 18182 (2010).

## ORR — oxygen reduction
Dispatch key: `{"ORR": "<pathway_name>"}`. Cathodic counterpart of OER. Same
\*OH–\*OOH scaling relation (~3.2 eV) caps activity at η ≈ 0.3–0.4 V.

Pathways implemented
--------------------
- ``'4e_associative'``  : O₂ → \*O₂ → \*OOH → \*O → \*OH → H₂O          (Pt, Pd, Pt₃Ni)
- ``'4e_dissociative'`` : O₂ → 2 \*O → \*OH → H₂O                       (Pt(111) at low T)
- ``'2e_to_h2o2'``      : O₂ → \*O₂ → \*OOH → H₂O₂(aq)                  (Au, Hg, Co–N₄, Fe–N₄)

References
----------
Nørskov et al. *J. Phys. Chem. B* **108**, 17886 (2004).
Stamenkovic et al. *Nat. Mater.* **6**, 241 (2007).
Greeley et al. *Nat. Chem.* **1**, 552 (2009).
Siahrostami et al. *Nat. Mater.* **12**, 1137 (2013).
Kulkarni et al. *Chem. Rev.* **118**, 2302 (2018).

## NRR — nitrogen reduction
Dispatch key: `{"NRR": "<pathway_name>"}`. Electrochemical analog of
Haber–Bosch. Strongly limited by HER competition and by weak \*N₂ binding on
most metals.

Pathways implemented
--------------------
- ``'distal'``       : \*N₂ → \*NNH → \*NNH₂ → \*N + NH₃ → \*NH → \*NH₂ → NH₃                 (Ru, Mo, Re, Mo–N₃)
- ``'alternating'``  : \*N₂ → \*NNH → \*NHNH → \*NHNH₂ → \*N₂H₄ → \*NH₂ + NH₃ → \*NH₃ → NH₃   (Fe, Co, Fe–N₄)
- ``'dissociative'`` : N₂ → 2 \*N → 2 \*NH → 2 \*NH₂ → 2 NH₃                                  (Fe, Ru, Os; needs strong \*N binding)

References
----------
Skúlason et al. *Phys. Chem. Chem. Phys.* **14**, 1235 (2012).
Montoya et al. *ChemSusChem* **8**, 2180 (2015).
Singh et al. *ACS Catal.* **7**, 706 (2017).
Andersen et al. *Nature* **570**, 504 (2019).
Suryanto et al. *Nat. Catal.* **2**, 290 (2019).

---

# Planned / candidate reactions

The following all fit the same `ReactionPathway` / `_create_adsorbate_with_dummy`
template and would be drop-in additions. Effort estimates are relative to the
HER block (smallest existing reaction; ~100 lines).

## HOR — hydrogen oxidation
Reverse of HER (Volmer + Heyrovsky/Tafel run backwards). Effort: tiny; just
reversed pathways, no new geometry. Useful for fuel-cell anode work.

References
----------
Sheng et al. *J. Electrochem. Soc.* **157**, B1529 (2010).
Strmcnik et al. *Nat. Chem.* **5**, 300 (2013).
Durst et al. *Energy Environ. Sci.* **7**, 2255 (2014).

## AOR — ammonia oxidation
2 NH₃ + 6 OH⁻ → N₂ + 6 H₂O + 6 e⁻. Dehydrogenation cascade
\*NH₃ → \*NH₂ → \*NH → \*N → N₂; reuses every intermediate already in NOXRR.
Effort: tiny. Important for direct-ammonia fuel cells.

References
----------
Vidal-Iglesias et al. *J. Catal.* **233**, 237 (2005).
Bunce & Bejan *Electrochim. Acta* **56**, 8085 (2011).
Boggs & Botte *J. Power Sources* **192**, 573 (2009).
Rosca et al. *Chem. Rev.* **109**, 2209 (2009).

## MOR — methanol oxidation
CH₃OH + H₂O → CO₂ + 6 H⁺ + 6 e⁻. Dehydrogenation through \*CH₃OH → \*CH₂OH →
\*CHOH → \*CO → CO₂, with the well-known \*CO poisoning detour. Reuses CO,
CHO, CHOH, CH₂O, CH₂OH from CO2RR. Effort: small. New ref: gas-phase CH₃OH
already present.

References
----------
Hamnett *Catal. Today* **38**, 445 (1997).
Yu et al. *J. Phys. Chem. C* **116**, 10906 (2012).
Cao et al. *Chem. Soc. Rev.* **35**, 1230 (2006).
Wasmus & Küver *J. Electroanal. Chem.* **461**, 14 (1999).

## FAOR — formic acid oxidation
HCOOH → CO₂ + 2 H⁺ + 2 e⁻. Direct (\*HCOO → CO₂) vs indirect (\*COOH → \*CO
poison → CO₂) bifurcation. Reuses OCHO/COOH/CO from CO2RR. Effort: small.
New ref: HCOOH crystal (look up MP).

References
----------
Capon & Parsons *J. Electroanal. Chem.* **44**, 1 (1973) — original dual-path.
Cuesta et al. *Phys. Chem. Chem. Phys.* **13**, 20091 (2011).
Yu & Pickup *J. Power Sources* **182**, 124 (2008).
Chen et al. *Angew. Chem. Int. Ed.* **45**, 981 (2006).

## BER — bromine evolution
2 Br⁻ → Br₂ + 2 e⁻. Direct analog of CER with Br substituted for Cl.
Volmer–Tafel, Volmer–Heyrovsky, and Krishtalik (\*OBr) pathways. Effort: small
(copy of CER). New refs: \*Br, \*OBr adsorbates and Br₂ crystal.

References
----------
Vos & Koper *J. Electroanal. Chem.* **819**, 260 (2018).
Karlsson & Cornell *Chem. Rev.* **116**, 2982 (2016).
Tomashov & Strukov *Russ. J. Phys. Chem.* **35**, 1112 (1961) — original kinetics.

## UOR — urea oxidation
CO(NH₂)₂ + 6 OH⁻ → N₂ + CO₂ + 5 H₂O + 6 e⁻. Direct vs Ni(III)-mediated
(indirect) mechanism. Effort: medium-large. Adds urea-specific intermediate
geometries (\*urea, \*NH₂CO, \*NHCO, …) and a urea gas-phase reference.
Practical relevance: wastewater + H₂ co-generation.

References
----------
Boggs et al. *Chem. Commun.* 4859 (2009).
Wang et al. *J. Mater. Chem. A* **5**, 3208 (2017).
Yan et al. *Appl. Catal. B* **122–123**, 73 (2012).
Vedharathinam & Botte *Electrochim. Acta* **108**, 660 (2013).

## EOR — ethanol oxidation
C₂H₅OH → CO₂ (full oxidation) or CH₃COOH (partial). C₁ vs C₂ branching with
C–C bond cleavage as the central question. Effort: large; the mechanism is
genuinely unsettled and the intermediate library is extensive (\*CH₃CH₂OH,
\*CH₃CHO, \*CH₃COO, \*CHCO, …).

References
----------
Lai et al. *ACS Catal.* **2**, 1042 (2012).
Bayer et al. *Appl. Catal. B* **138–139**, 313 (2013).
Ferrin & Mavrikakis *J. Am. Chem. Soc.* **131**, 14381 (2009).
Wang et al. *J. Phys. Chem. C* **116**, 6675 (2012).

## Photo-driven variants
Photocatalytic water splitting (HER + OER on a semiconductor), CO₂
photoreduction, N₂ photofixation. The pathway machinery is identical to the
electrochemical counterparts; the photo aspect enters as a thermodynamic gate
(semiconductor band edges vs intermediate energies) and belongs in the
analysis layer downstream of `adsorbates.py`. Likely no new code in
`adsorbates.py` — just a flag downstream.

References
----------
Walter et al. *Chem. Rev.* **110**, 6446 (2010).
Fujishima & Honda *Nature* **238**, 37 (1972).
White et al. *Chem. Rev.* **115**, 12888 (2015) — CO₂ photoreduction.
Comer et al. *Joule* **3**, 1578 (2019) — N₂ photofixation.

---

# Overpotential calculators

One module per reaction under `workchains/`. Each implements the CHE
model with the same return shape:

```python
overpotential, dg_steps, dg_cumulative = calculate_<rxn>_overpotential(
    adsorption_energies, pathway_name, method, func)
```

where `adsorption_energies` is a dict keyed by `*X` adsorbate names,
`pathway_name` matches the pathway keys in this document, and
`(method, func)` pick the gas-phase reference set from
`workchains/references.yaml` (e.g. `("dft", "r2SCAN")`,
`("uPET", "r2SCAN")`).

| Reaction | Module                  | Pathway keys                                       |
|----------|-------------------------|----------------------------------------------------|
| OER      | `workchains/oer.py`     | (4e- mechanism, fixed)                             |
| CO2RR    | `workchains/co2rr.py`   | `co2_to_co`, `co2_to_hcooh`, `co_to_ch4`, `co_to_ch3oh`, `co2_to_ch4`, `co2_to_ch3oh`, `co2_to_c2h4` |
| NOXRR    | `workchains/noxrr.py`   | `no_dissociative`, `no_to_nh3_noh`, `no_to_nh3_nhoh`, `no_to_n2o`, `no2_to_no`, `no3_to_nh3`, `no3_to_n2` |
| CER      | `workchains/cer.py`     | `volmer_tafel`, `volmer_heyrovsky`, `krishtalik`   |
| HER      | `workchains/her.py`     | `volmer_tafel`, `volmer_heyrovsky`                 |
| ORR      | `workchains/orr.py`     | `4e_associative`, `4e_dissociative`, `2e_to_h2o2`  |
| NRR      | `workchains/nrr.py`     | `distal`, `alternating`, `dissociative`            |

**ZPE corrections** are loaded once at module import via
`load_zpe(reaction)` from `workchains/zpe_corrections.yaml`. That file
covers every adsorbate and gas-phase species touched by these pathways;
see the file header for the convention (full harmonic ZPE per species,
not a delta against ZPE-folded references).

**Gas-phase references** are pulled per-call via
`load_references(method, func)` from `workchains/references.yaml`.
Prerequisites flagged by the new modules:

- `cer.py` requires a `Cl2:` entry in every `(method, func)` block.
- `orr.py`'s `2e_to_h2o2` pathway requires an `H2O2:` entry.

The CER `equilibrium_potential = 1.36` is vs SHE (the Cl⁻/Cl₂ couple),
not vs RHE — downstream code reading this value should account for the
reference difference.

The pathway dicts in the per-reaction modules use the same stoichiometric
convention as `co2rr.py` (positive = product, negative = reactant; each
electrochemical step adds `H2: +1/2` for proton-coupled chemistry or
`Cl2: +1/2` for CER's chloride-coupled chemistry).

---

# Nano particles
Dispatch key: `'nano_particles': False / True`.

The key has to be present or the frontend update will fail. With `True`, the
catalysis WorkChain is bypassed entirely and nano-particle generation runs
instead. More documentation will follow.
