# `precursor_search` — containerized synthesis-route literature search

An opt-in stage of `SynthesizabilityWorkChain` that launches a **containerized
agent** to web-search recent publications for *proven* ways to synthesize a
target material, and returns the **DOIs** and **synthesis paths** it finds for
further processing.

## Pieces

| File | Role |
|------|------|
| `calculation.py` → `PrecursorSearchCalculation` | AiiDA CalcJob (entry point `precursor_search`). Stages the request JSON, runs the container code, retrieves the response JSON. Container-engine agnostic. |
| `parser.py` → `PrecursorSearchParser` | Parser (entry point `precursor_search_parser`). Wraps `output.json` into the `output_dict` Dict node. |
| `../files/precursor_agent.py` | **Reference** agent = the I/O contract + a stdlib-only stub for testing. Replace with the real container. |
| `container/` | **Real containerized agent** (debian-slim, rootless/udocker): `Dockerfile` + `precursor-agent` wrapper (Claude Code headless WebSearch/WebFetch) + `build.sh`/`test_plumbing.sh`. Drop-in for the stub; same `output.json` contract. See `container/README.md`. |

The workchain wiring lives in `workchains/synthesizability.py`
(`should_search_precursors` / `run_precursor_search` / `inspect_precursor_search`),
modeled on the existing remote-SQS step (`run_sqs`).

## Container contract

The CalcJob runs (cmdline overridable via `parameters['cmdline_params']`):

```
<container entrypoint> --request=request.json --output=output.json
```

**Input — `request.json`** (staged into the run dir):

```json
{
  "chemical_formula": "Y2Ru2O7",
  "reduced_formula": "Y2Ru2O7",
  "elements": ["Y", "Ru", "O"],
  "candidates": [
    {"uuid": "…", "formula": "Y2Ru2O7", "label": "synthesizable",
     "score": 0.81, "ehull_eV_per_atom": 0.0}
  ],
  "max_results": 20,
  "since_year": 2015,
  "include_preprints": true,
  "methods": ["solid-state", "sol-gel", "hydrothermal", "flux", "cvd", "precipitation"]
}
```

**Output — `output.json`** (written by the agent in the run dir, then retrieved):

```json
{
  "formula": "Y2Ru2O7",
  "results": [
    {
      "doi": "10.1021/jacs.xxxxx",
      "title": "…",
      "year": 2021,
      "url": "https://doi.org/10.1021/jacs.xxxxx",
      "synthesis_routes": [
        {
          "method": "solid-state",
          "precursors": ["Y2O3", "RuO2"],
          "steps": ["mix stoichiometric powders", "calcine 1000 °C / 24 h / air"],
          "conditions": {"temperature_C": 1000, "time_h": 24, "atmosphere": "air"},
          "product": "Y2Ru2O7",
          "confidence": 0.82,
          "evidence": "verbatim passage supporting the route"
        }
      ]
    }
  ],
  "n_results": 1,
  "agent": {"model": "…", "search_provider": "…", "version": "…"}
}
```

The parser requires a top-level object with a `results` **list**; everything
else is pass-through, so you can add fields without touching the plugin. Each
result *should* carry a `doi` and a `synthesis_routes` list. **Never fabricate
DOIs** — return `results: []` when nothing is found.

## Where results land

- Workchain output node `precursor_search` (Dict).
- `DBComposition.attributes["precursor_search"]` = the full payload, for
  downstream processing.

## Enabling

`input.yaml`:

```yaml
synthesizability:
  enabled: true
  precursor_search:
    enabled: true
    only_synthesizable: true   # search only compositions with a synthesizable candidate
    max_results: 20
    since_year: 2015
```

`config.yaml` → `codes.precursor_search.code_string` must point at the registered
container code (e.g. `precursor_agent@<computer>`).

## Deploy notes

- After adding the plugin, re-register the entry points so AiiDA sees them:
  `pip install -e .` (or `verdi plugin list aiida.calculations | grep precursor_search`).
- Register the image as a `core.code.containerized` `ContainerizedCode`. Its
  `filepath_executable` is the **in-container** path `/usr/local/bin/precursor-agent`
  (AiiDA does not check it against the remote filesystem for containerized codes);
  the `engine_command` (with `{image_name}`) wraps it and the CalcJob appends only
  `--request=… --output=…`.
- The compute node needs **outbound network** (HTTPS to the Anthropic API only).
- **API key:** it must reach the *container on the node* — `input.yaml` /
  `config.yaml` are submit-side only and will **not** deliver it. Keep the key in a
  0600 file on the node and inject it via the engine command's `--env-file`
  (udocker) or a Code `prepend_text` that `cat`s it (apptainer). **Never** put it
  in a CalcJob input or `metadata.options.environment_variables` — those land in
  the provenance graph. Full recipe: [`container/README.md`](container/README.md)
  → *Secrets & network*.
