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

- Provide `ANTHROPIC_API_KEY` (or `CLAUDE_CODE_OAUTH_TOKEN`) via the Computer's
  `environment_variables` / `prepend_text` — **never** a CalcJob input, so it
  stays out of the provenance graph. Optional: `ANTHROPIC_MODEL`,
  `ANTHROPIC_BASE_URL`.
- WebSearch/WebFetch run **server-side at Anthropic**, so the only egress the
  compute node needs is HTTPS to the Anthropic API — not the whole web.

## Tests

- `./test_plumbing.sh` — offline; stubs `claude` and asserts the output contract
  (clean JSON, fenced JSON, model garbage, missing request). No key needed.
- Live end-to-end: set `ANTHROPIC_API_KEY` and run
  `precursor-agent --request=examples/request.json --output=output.json`.

## Tuning

`precursor-agent` env knobs: `ANTHROPIC_MODEL` (pin the model). The agent is
restricted to the `WebSearch`/`WebFetch` tools under default permission mode
(no filesystem access). To widen methods/recency, edit the request, not the image.
