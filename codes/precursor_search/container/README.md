# precursor-search agent container

The real container behind `uvsib`'s `PrecursorSearchCalculation` — a drop-in
replacement for the reference stub `uvsib/codes/files/precursor_agent.py`. It
web-searches recent literature for **proven** synthesis routes of a target
inorganic compound and returns the DOIs + routes as JSON for the
`SynthesizabilityWorkChain` synth module.

Runs **rootless under udocker** on remote machines. Minimal `debian-slim` image:
glibc + `ca-certificates` + `jq` + the self-contained Claude Code native binary.
No Node/Python/git/toolchain — the agent only searches and summarizes.
**~310 MB** uncompressed (the ~234 MB `claude` binary dominates).

## Contract (unchanged from the stub)

```
precursor-agent --request=request.json --output=output.json
```

- **Input** `request.json`: `chemical_formula`, `reduced_formula`, `elements`,
  `candidates`, `max_results`, `since_year`, `include_preprints`, `methods`.
- **Output** `output.json`: a dict with a `results` **list** (what
  `PrecursorSearchParser` requires), plus `formula`, `query` (the echoed
  request, for provenance), `n_results`, and `agent` (`model`,
  `search_provider`, `version`, optional `error`). Each result carries `doi`,
  `title`, `year`, `url`, and a `synthesis_routes` list (`method`, `precursors`,
  `steps`, `conditions{temperature_C,time_h,atmosphere}`, `product`,
  `confidence`, `evidence`). See `examples/request.json`.

The **wrapper owns all file I/O** and always writes a schema-valid dict — even if
the model fails or returns prose — so the parser never hits
`ERROR_INVALID_OUTPUT`. **DOIs are never fabricated**: no verifiable route ⇒
`results: []`.

## Build & deploy

```bash
./build.sh           # podman/docker build -> tarball -> udocker import
```

Register the imported image as an AiiDA `ContainerizedCode`; set
`codes.precursor_search.code_string` to it. The CalcJob appends
`--request=… --output=…`.

## Secrets & network

The `ANTHROPIC_API_KEY` (or `CLAUDE_CODE_OAUTH_TOKEN`) must reach the **container
on the remote compute node** — it is *not* a submit-side value, so putting it in
`input.yaml` or `config.yaml` does nothing (those are read on the submitter; the
agent runs in the container on the node). Inject it at job runtime through the
registered `ContainerizedCode`'s **engine command**, reading a private file that
lives only on the node:

1. On the compute node, store the key in a 0600 env-file (never committed):

   ```bash
   mkdir -p ~/.config/anthropic && umask 077
   printf 'ANTHROPIC_API_KEY=sk-ant-...\n' >  ~/.config/anthropic/agent.env
   # optional, one per line:  ANTHROPIC_MODEL=claude-...   ANTHROPIC_BASE_URL=...
   chmod 600 ~/.config/anthropic/agent.env
   ```

2. Register the image as a containerized code whose engine command passes that
   file into the container with `--env-file` (the value never enters `argv`):

   ```bash
   verdi code create core.code.containerized \
     --label precursor_agent --computer <remote> \
     --filepath-executable /usr/local/bin/precursor-agent \
     --image-name precursor-agent \
     --engine-command 'udocker run --rm --env-file=$HOME/.config/anthropic/agent.env -v "$PWD":"$PWD" -w "$PWD" {image_name}'
   ```

   `$HOME`/`$PWD` expand in the job script at run time. AiiDA appends the
   `filepath_executable` and then the CalcJob's `--request=… --output=…`.

**Why this is safe:** the key is *not* baked into the image, *not* a CalcJob input
(so it never enters the provenance graph), *not* stored in the AiiDA DB (only the
env-file **path** is, inside the engine command), and *not* visible in `ps`/`argv`
(`--env-file`, not `-e KEY=value`). It exists only as a 0600 file on the node. The
sole network egress needed is HTTPS to the Anthropic API — WebSearch/WebFetch run
**server-side at Anthropic**, not from the node.

> Do **not** use the CalcJob `metadata.options.environment_variables` for the key:
> that value *is* serialized onto the CalcJobNode (provenance leak). `--env-file`
> (above) or a Code/Computer `prepend_text` that `cat`s the file are the safe
> vehicles. For apptainer/singularity (no `--env-file`), use `prepend_text`:
> `export ANTHROPIC_API_KEY="$(cat ~/.config/anthropic/key)"` and `apptainer exec
> {image_name}` (host env passes through; the DB stores only the `cat` command).

## Tests

- `./test_plumbing.sh` — offline; stubs `claude` and asserts the output contract
  (clean JSON, fenced JSON, model garbage, missing request). No key needed.
- Live end-to-end: set `ANTHROPIC_API_KEY` and run
  `precursor-agent --request=examples/request.json --output=output.json`.

## Tuning

`precursor-agent` env knobs: `ANTHROPIC_MODEL` (pin the model). The agent is
restricted to the `WebSearch`/`WebFetch` tools under default permission mode
(no filesystem access). To widen methods/recency, edit the request, not the image.
