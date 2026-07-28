"""Remote (CPU-queue) vacancy-ordering enumeration for the battery pathway.

The Ewald-ranked enumeration over up-to-``supercell_max_atoms`` supercells is
far too CPU-heavy for the AiiDA daemon host, so BatteryWorkChain ships it to
the code named by input.yaml ``battery: enum_code:`` (default ``sqs_cpu`` --
the same CPU queue the synthesizability finite_T SQS generator uses).

Staged into the job as ``aiida.py`` next to
  battery_enum.py                the pure enumeration module, shipped
                                 VERBATIM from workchains/battery_enum.py via
                                 the file namespace (single canonical copy)
  battery_enum_request.json      the request (--request cmdline arg)

Request format:
    {"working_ion": "Li",
     "params": {"n_x_steps": ..., "max_configs_per_x": ...,
                "supercell_max_atoms": ...},
     "hosts": [{"uuid": <bulk uuid>, "structure": <pmg dict>}, ...]}

Output (output.json, generic sqs_parser -> Dict):
    {"hosts": [{"uuid": ..., "n_sites": ..., "counts": [...],
                "ewald_ranked": ..., "configs": [<pmg dict>, ...]}, ...],
     "failed": [{"uuid": ..., "reason": ...}, ...]}

The per-host uuid is ECHOED so the workchain attributes configurations by
uuid, never by position (a failed host must not shift the mapping). A host
that raises is recorded in ``failed`` and the job keeps going.
"""
import argparse
import json
import traceback

import battery_enum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=str,
                        default="battery_enum_request.json")
    args = parser.parse_args()

    with open(args.request, "r", encoding="utf-8") as f:
        request = json.load(f)
    params = request["params"]
    working_ion = request["working_ion"]

    hosts_out, failed = [], []
    for host in request["hosts"]:
        uuid = host.get("uuid")
        try:
            plan = battery_enum.enumerate_deintercalation(
                host["structure"], working_ion,
                n_x_steps=params["n_x_steps"],
                max_configs_per_x=params["max_configs_per_x"],
                supercell_max_atoms=params["supercell_max_atoms"])
            configs = [s.as_dict() for k in plan["counts"]
                       for s in plan["configs"][k]]
            hosts_out.append({
                "uuid": uuid,
                "n_sites": plan["n_sites"],
                "counts": list(plan["counts"]),
                "ewald_ranked": bool(plan["ewald_ranked"]),
                "configs": configs,
            })
            print(f"host {uuid}: N={plan['n_sites']} ion sites, grid "
                  f"{plan['counts']}, {len(configs)} configs", flush=True)
        except Exception as exc:  # noqa: BLE001 -- record + continue per host
            failed.append({"uuid": uuid,
                           "reason": f"{type(exc).__name__}: {exc}"})
            print(f"host {uuid} FAILED: {exc}", flush=True)
            traceback.print_exc()

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump({"hosts": hosts_out, "failed": failed}, f)
    print(f"wrote output.json: {len(hosts_out)} host(s) enumerated, "
          f"{len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
