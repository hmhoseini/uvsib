"""Adsorbates on NANO-PARTICLES: site finding, placement, relaxation.

Implements Route 1 of ``docs/nano_particle_path.md``: pymatgen's
AdsorbateSiteFinder detects sites by z-height and reorients onto the global
+z axis, which is meaningless for a free particle -- so sites are found from
LOCAL geometry instead:

  * surface atoms  : coordination number (covalent-radii bond graph) below a
                     threshold (FCC bulk = 12; default surface cutoff CN <= 9)
  * outward normal : n_i = normalize(r_i - mean(r_neighbors(i))) -- the
                     coordination shell sits on the material side, so this
                     points outward for convex AND concave regions
  * sites          : ontop  (r_i, n_i)
                     bridge (midpoint of a bonded surface pair, n_i + n_j)
                     hollow (centroid of a bonded surface triangle, sum n)
  * dedup + cap    : near-duplicate sites are removed by a local-environment
                     fingerprint; the remainder is capped per site type by
                     farthest-point sampling so a symmetric particle does not
                     burn hundreds of equivalent relaxations.

Everything chemical is REUSED from the slab runner, which is staged alongside
as ``slab_adsorbates.py`` (= codes/files/adsorbates.py): the X-dummy adsorbate
registry and reaction pathways (generate_*_adsorbates), gas-phase references,
``has_reasonable_distances`` and the post-relaxation validator
(``validate_relaxed_adsorbate`` -- "adsorbate = last n atoms" holds here too).
Placement replicates ``asf.add_adsorbate(..., translate=False, reorient=True)``
for an arbitrary normal: rotate the molecule's internal +z onto the site
normal (Rodrigues), put the X dummy on the site, drop X. On a clash
(has_reasonable_distances fails) the same site is retried with the molecule
rotated azimuthally by 90/180/270 degrees before the set is discarded.

The bare particle is relaxed HERE (positions only, no cell filter, no
FixAtoms -- a free particle may restructure) and its energy is the '*' entry
of every energy set, so clean/adsorbed/gas references all come from the same
model in the same job.

Input (``input_structures.json``): [{"uuid": <str>, "structure": <pymatgen
Structure.as_dict() -- particle centered in a vacuum box>}, ...] -- a BATCH
of particles (the workchain submits batches of DBNanoParticles rows; the
model and the gas-phase references are loaded/relaxed once per job). One
failing particle is reported in its result entry and does NOT abort the
rest of the batch.

Output (``output.json``, sqs_parser-compatible):
    {"results": [
        {"uuid":       <particle uuid>,
         "particle":   {"uuid": ..., "energy": <bare relaxed E>,
                        "structure": <ase jsonio of the relaxed bare particle>},
         "structures": [<relaxed_set>, ...]},  # SAME set contract as the slab
                                               # runner: site_type, ads_coord,
                                               # repeat, structures=[adsorbates
                                               # ..., gas refs..., bare
                                               # particle('*')]
        {"uuid": ..., "error": "<why this particle failed>"},   # failure entry
        ...]}
Plus total.txt / failed.txt / rejected.json bookkeeping, like the slab runner.
"""
import argparse
import json

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import jsonio
from ase.optimize.bfgslinesearch import BFGSLineSearch
from pymatgen.core import Structure

import slab_adsorbates as sa

# metal-metal bond cutoff for the particle's coordination graph; same factor
# as the adsorbate bond graphs so the two geometrical layers agree
BOND_TOL = 1.25
AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)


# --------------------------------------------------------------------------- #
# cluster surface geometry (docs/nano_particle_path.md, Suggestion 2, Route 1)
# --------------------------------------------------------------------------- #
def _bond_neighbors(atoms):
    """Neighbor index lists from the covalent-radii bond criterion (no PBC:
    the particle sits centered in a vacuum box, plain distances suffice)."""
    pos = atoms.get_positions()
    radii = np.array([covalent_radii[z] for z in atoms.get_atomic_numbers()])
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    cut = BOND_TOL * (radii[:, None] + radii[None, :])
    np.fill_diagonal(dist, np.inf)
    bonded = dist <= cut
    return [np.where(bonded[i])[0] for i in range(len(atoms))], bonded


def find_sites(atoms, surface_cn_max, min_normal=0.25):
    """Ontop / bridge / hollow sites with local outward normals.

    Returns a list of {"site_type", "position", "normal", "anchors"} dicts in
    a deterministic order. Sites whose normal points inward relative to the
    particle centroid are dropped (spurious sites inside concavities/holes).
    """
    pos = atoms.get_positions()
    centroid = pos.mean(axis=0)
    neigh, bonded = _bond_neighbors(atoms)

    surface, normals = [], {}
    for i in range(len(atoms)):
        if len(neigh[i]) == 0 or len(neigh[i]) > surface_cn_max:
            continue
        n = pos[i] - pos[neigh[i]].mean(axis=0)
        if np.linalg.norm(n) < min_normal:      # degenerate shell: fall back
            n = pos[i] - centroid               # to the radial direction
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        surface.append(i)
        normals[i] = n / norm

    def outward(p, n):
        r = p - centroid
        return np.linalg.norm(r) < 1e-6 or float(np.dot(n, r)) > 0.0

    sites = []
    for i in surface:                                        # ontop
        if outward(pos[i], normals[i]):
            sites.append({"site_type": "ontop", "position": pos[i].copy(),
                          "normal": normals[i], "anchors": [int(i)]})
    for a, i in enumerate(surface):                          # bridge
        for j in surface[a + 1:]:
            if not bonded[i, j]:
                continue
            n = normals[i] + normals[j]
            norm = np.linalg.norm(n)
            p = 0.5 * (pos[i] + pos[j])
            if norm > 1e-6 and outward(p, n / norm):
                sites.append({"site_type": "bridge", "position": p,
                              "normal": n / norm, "anchors": [int(i), int(j)]})
    for a, i in enumerate(surface):                          # hollow
        for b in range(a + 1, len(surface)):
            j = surface[b]
            if not bonded[i, j]:
                continue
            for k in surface[b + 1:]:
                if not (bonded[i, k] and bonded[j, k]):
                    continue
                n = normals[i] + normals[j] + normals[k]
                norm = np.linalg.norm(n)
                p = (pos[i] + pos[j] + pos[k]) / 3.0
                if norm > 1e-6 and outward(p, n / norm):
                    sites.append({"site_type": "hollow", "position": p,
                                  "normal": n / norm,
                                  "anchors": [int(i), int(j), int(k)]})
    return sites


def dedup_sites(atoms, sites, shell=5.0, decimals=1):
    """Drop near-duplicate sites: fingerprint = site type + the sorted
    (element, distance) environment within ``shell`` A of the site position.
    Symmetry-equivalent sites of an ordered particle collapse to one."""
    pos = atoms.get_positions()
    nums = atoms.get_atomic_numbers()
    seen, kept = set(), []
    for s in sites:
        d = np.linalg.norm(pos - s["position"], axis=1)
        m = d <= shell
        fp = (s["site_type"],
              tuple(sorted(zip(nums[m].tolist(), np.round(d[m], decimals).tolist()))))
        if fp in seen:
            continue
        seen.add(fp)
        kept.append(s)
    return kept


def select_sites(sites, max_sites):
    """Cap to ``max_sites``, split evenly over the site types present, picking
    spatially spread representatives (greedy farthest-point, deterministic)."""
    if max_sites <= 0 or len(sites) <= max_sites:
        return sites
    by_type = {}
    for s in sites:
        by_type.setdefault(s["site_type"], []).append(s)
    quota, extra = divmod(max_sites, len(by_type))
    selected = []
    for t_idx, t in enumerate(sorted(by_type)):
        want = quota + (1 if t_idx < extra else 0)
        pool = by_type[t]
        if len(pool) <= want:
            selected.extend(pool)
            continue
        chosen = [pool[0]]
        rest = pool[1:]
        while len(chosen) < want:
            dmin = [min(np.linalg.norm(c["position"] - r["position"])
                        for c in chosen) for r in rest]
            k = int(np.argmax(dmin))
            chosen.append(rest.pop(k))
        selected.extend(chosen)
    return selected


# --------------------------------------------------------------------------- #
# placement: molecule internal +z -> site normal, X dummy on the site
# --------------------------------------------------------------------------- #
def _rotation_to(normal):
    """Rotation matrix mapping [0, 0, 1] onto ``normal`` (Rodrigues)."""
    ez = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(ez, normal))
    if c > 1.0 - 1e-10:
        return np.eye(3)
    if c < -1.0 + 1e-10:                       # antiparallel: flip about x
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(ez, normal)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def place_adsorbate(particle, mol, position, normal, azimuth_deg=0.0):
    """Particle + adsorbate Atoms: replicates the slab convention
    (asf.add_adsorbate with translate=False, reorient=True) for a local
    normal. The adsorbate is APPENDED (validator: adsorbate = last n atoms)."""
    az = np.deg2rad(azimuth_deg)
    raz = np.array([[np.cos(az), -np.sin(az), 0.0],
                    [np.sin(az), np.cos(az), 0.0],
                    [0.0, 0.0, 1.0]])
    rot = _rotation_to(np.asarray(normal, dtype=float))
    symbols, coords = [], []
    for site in mol:
        if str(site.specie.symbol) == "X":
            continue
        symbols.append(str(site.specie.symbol))
        coords.append(rot @ (raz @ np.asarray(site.coords, dtype=float)))
    ads = Atoms(symbols=symbols, positions=np.array(coords) + position)
    combined = particle.copy()
    combined.calc = None
    combined += ads
    combined.info = dict(particle.info)
    combined.info["adsorbate"] = mol.properties["adsorbate"]
    return combined, len(ads)


def build_adsorbates(reaction, pathway_name):
    """Adsorbate Molecules + pathway object from the SHARED slab registry."""
    if reaction == "OER":
        return sa.generate_oer_adsorbates(), None
    generators = {"CO2RR": sa.generate_co2rr_adsorbates,
                  "NOXRR": sa.generate_noxrr_adsorbates,
                  "CER": sa.generate_cer_adsorbates,
                  "HER": sa.generate_her_adsorbates,
                  "ORR": sa.generate_orr_adsorbates,
                  "NRR": sa.generate_nrr_adsorbates}
    if reaction not in generators:
        raise ValueError(f"Unknown reaction: {reaction}. "
                         f"Expected one of: OER, {', '.join(sorted(generators))}")
    pathway_obj, ads_dict = generators[reaction](pathway_name)
    return list(ads_dict.values()), pathway_obj


# --------------------------------------------------------------------------- #
# main flow
# --------------------------------------------------------------------------- #
def _process_particle(entry, calc, adsorbates, expected_graphs,
                      relaxed_refs_json, model_key, fmax, max_steps,
                      max_sites, surface_cn_max, validate_adsorbates,
                      stats, rejected):
    """Bare relax + sites + adsorbate relaxations for ONE particle of the
    batch. Raises on unusable particles (caught by the batch loop)."""
    particle = sa.pmg_to_ase(Structure.from_dict(entry["structure"]))

    # bare particle: positions-only relax, no constraints (free restructuring)
    particle.calc = calc
    relax = BFGSLineSearch(particle, maxstep=0.1, logfile="opt.log")
    relax.run(fmax=fmax, steps=max_steps)
    if not relax.converged:
        raise RuntimeError(f"bare particle did not converge in {max_steps} steps")
    particle_energy = particle.get_potential_energy()
    particle.info["adsorbate"] = "*"
    particle.info[model_key] = particle_energy
    stats["total"] += 1

    # sites on the RELAXED particle
    sites = find_sites(particle, surface_cn_max)
    n_raw = len(sites)
    sites = select_sites(dedup_sites(particle, sites), max_sites)
    if not sites:
        raise RuntimeError(f"no adsorption sites found "
                           f"(surface_cn_max={surface_cn_max})")
    counts = {t: sum(1 for s in sites if s["site_type"] == t)
              for t in ("ontop", "bridge", "hollow")}
    print(f"particle {entry['uuid']}: {len(particle)} atoms, {n_raw} raw sites "
          f"-> {len(sites)} after dedup/cap {counts}")

    relaxed_sets = []
    for site in sites:
        relaxed_structures = []
        for ads in adsorbates:
            name = ads.properties["adsorbate"]
            candidate = None
            for az in AZIMUTHS_DEG:            # clash retries about the normal
                trial, _ = place_adsorbate(particle, ads, site["position"],
                                           site["normal"], az)
                if sa.has_reasonable_distances(trial):
                    candidate = trial
                    break
            if candidate is None:
                rejected.append({"uuid": entry["uuid"], "adsorbate": name,
                                 "site_type": site["site_type"],
                                 "ads_coord": site["position"].tolist(),
                                 "reason": "no clash-free placement (all azimuths)"})
                stats["failed"] += 1
                break

            candidate.calc = calc
            relax = BFGSLineSearch(candidate, maxstep=0.1, logfile="opt.log")
            try:
                relax.run(fmax=fmax, steps=max_steps)
            except Exception as e:  # noqa: BLE001 -- count + drop the set
                print(f"Warning: relaxation failed for {name}: {e}")
                stats["failed"] += 1
                break
            if not relax.converged:
                stats["failed"] += 1
                break
            ads_energy = candidate.get_potential_energy()
            candidate.info[model_key] = ads_energy

            if validate_adsorbates:
                verdict = sa.validate_relaxed_adsorbate(
                    candidate, expected_graphs[name].number_of_nodes(),
                    expected_graphs[name], ads_energy)
                if not verdict.ok:
                    rejected.append({"uuid": entry["uuid"], "adsorbate": name,
                                     "site_type": site["site_type"],
                                     "ads_coord": site["position"].tolist(),
                                     "reason": verdict.reason})
                    stats["failed"] += 1
                    break

            relaxed_structures.append(jsonio.encode(candidate))
            stats["total"] += 1

        if len(relaxed_structures) == len(adsorbates):
            relaxed_sets.append({
                "site_type": site["site_type"],
                "ads_coord": site["position"].tolist(),
                "site_anchors": site["anchors"],
                "repeat": [1, 1, 1],
                "structures": (relaxed_structures + relaxed_refs_json
                               + [jsonio.encode(particle)]),
            })

    print(f"particle {entry['uuid']}: {len(relaxed_sets)}/{len(sites)} "
          f"complete sets")
    return {
        "uuid": entry["uuid"],
        "particle": {"uuid": entry["uuid"], "energy": particle_energy,
                     "structure": jsonio.encode(particle)},
        "structures": relaxed_sets,
    }


def run_particle_adsorbates(ml_model, calc, fmax, max_steps, reaction, pathway,
                            max_sites=24, surface_cn_max=9,
                            validate_adsorbates=True):
    model_key = f"{ml_model.lower()}_energy"
    with open("input_structures.json") as f:
        entries = json.load(f)
    if not entries:
        raise RuntimeError("empty particle batch")

    adsorbates, pathway_obj = build_adsorbates(reaction, pathway)
    expected_graphs = {ads.properties["adsorbate"]: sa._adsorbate_reference_graph(ads)
                       for ads in adsorbates}

    stats = {"total": 0, "failed": 0}
    rejected = []

    # gas-phase references, relaxed ONCE for the whole batch (identical to the
    # slab runner); without them nothing downstream works -> hard failure
    relaxed_refs_json = []
    for name in sa._pathway_required_refs(reaction, pathway_obj):
        ref = sa.pmg_to_ase(sa._GAS_REF_REGISTRY[name])
        ref.info["adsorbate"] = name
        ref.calc = calc
        relax = BFGSLineSearch(ref, maxstep=0.1, logfile="opt.log")
        relax.run(fmax=fmax, steps=max_steps)
        if not relax.converged:
            raise RuntimeError(f"gas-phase reference '{name}' did not converge")
        ref.info[model_key] = ref.get_potential_energy() / sa._reference_molecule_count(ref, name)
        relaxed_refs_json.append(jsonio.encode(ref))
        stats["total"] += 1

    # one bad particle must not take down the other members of the batch:
    # failures become error entries in the output, visible to the workchain
    results = []
    for entry in entries:
        try:
            results.append(_process_particle(
                entry, calc, adsorbates, expected_graphs, relaxed_refs_json,
                model_key, fmax, max_steps, max_sites, surface_cn_max,
                validate_adsorbates, stats, rejected))
        except Exception as e:  # noqa: BLE001 -- report + continue the batch
            print(f"ERROR: particle {entry['uuid']} failed: {e}")
            stats["failed"] += 1
            results.append({"uuid": entry["uuid"], "error": str(e)})

    with open("output.json", "w") as f:
        json.dump({"results": results}, f)
    with open("total.txt", "w") as f:
        f.write(str(stats["total"]))
    with open("failed.txt", "w") as f:
        f.write(str(stats["failed"]))
    with open("rejected.json", "w") as f:
        json.dump(rejected, f)
    n_err = sum(1 for r in results if "error" in r)
    print(f"batch done: {len(results)} particle(s), {n_err} errored, "
          f"{stats['failed']} failed/rejected relaxations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ML_model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--device", type=str)
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--reaction", type=str)
    parser.add_argument("--pathway", type=str)
    parser.add_argument("--max_sites", type=int, default=24)
    parser.add_argument("--surface_cn_max", type=int, default=9)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    from _calculators import make_calculator
    calc = make_calculator(args.ML_model, model=args.model,
                           model_path=args.model_path, device=args.device,
                           task_name=args.task_name)

    run_particle_adsorbates(
        ml_model=args.ML_model, calc=calc, fmax=args.fmax,
        max_steps=args.max_steps, reaction=args.reaction, pathway=args.pathway,
        max_sites=args.max_sites, surface_cn_max=args.surface_cn_max,
        validate_adsorbates=not args.no_validate)
