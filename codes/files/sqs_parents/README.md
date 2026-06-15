# SQS parent structures

Parent unit cells for the SQS WorkChain (`workchains/sqs.py`). Each entry
should ship both a primitive cell (smallest cell carrying the full symmetry,
fastest for `icet.ClusterSpace`) and the conventional cell (easier to read /
verify by eye).

## Y2Ru2O7 -- pyrochlore A2B2O7

- `Y2Ru2O7_pyrochlore_primitive.cif`  -- 22 atoms, rhombohedral setting
- `Y2Ru2O7_pyrochlore_conventional.cif` -- 88 atoms, cubic Fd-3m (a = 10.262 A)
- Source: Materials Project entry [mp-20643](https://materialsproject.org/materials/mp-20643), space group Fd-3m (#227)
- MP-reported energetics: E_f = -2.562 eV/atom, E_above_hull = 0.030 eV/atom
- Cation sublattices for an SQS request: A = `"Y"` (8 sites per conv. cell),
  B = `"Ru"` (8 sites per conv. cell); O occupies the remaining 56 sites
  (kept fixed unless the request also defines an O sublattice).
