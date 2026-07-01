#!/usr/bin/env python3
"""Reference precursor-search agent -- the *contract* for the real container.

This is the executable that ``PrecursorSearchCalculation`` runs. In production
it is replaced by the full containerized agent you will provide; this reference
implementation exists so the contract is unambiguous and the AiiDA CalcJob +
parser + workchain wiring can be smoke-tested end to end *without* network
access or the real image.

Contract
--------
    python precursor_agent.py --request request.json --output output.json

Reads the request (see ``PrecursorSearchCalculation`` docstring), performs a web
search of recent publications for proven synthesis routes, and writes
``output.json`` in the current working directory.

To turn this into the real agent, replace ``run_agent`` with the body that does
the actual web search + extraction (e.g. query a literature API such as
Crossref / OpenAlex / Semantic Scholar, then have an LLM extract precursors,
steps and conditions, grounding every route in a DOI). Keep the input/output
JSON shape identical so the CalcJob and parser need no changes. Read any API
keys from the environment (``os.environ``); they are injected by the remote
Computer / image, never by the CalcJob inputs.
"""

import os
import sys
import json
import argparse


def run_agent(request):
    """Return the response payload for ``request``.

    REFERENCE STUB: returns an empty result set (no fabricated DOIs). Replace the
    body with the real web search + extraction. Must return a dict shaped like
    the ``output.json`` contract documented in the CalcJob.
    """
    formula = request.get("chemical_formula") or request.get("reduced_formula")

    # ---- plug the real literature/web-search agent in here -------------------
    # results = search_publications(request)          # -> list of {doi, ...}
    # results = extract_synthesis_routes(results)     # -> add synthesis_routes
    results = []  # stub: no routes; never invent DOIs
    # --------------------------------------------------------------------------

    return {
        "formula": formula,
        "query": request,
        "results": results,
        "n_results": len(results),
        "agent": {
            "model": None,
            "search_provider": None,
            "version": "reference-stub-1",
            "note": "reference stub -- replace with the containerized agent",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Precursor-search agent")
    parser.add_argument("--request", default="request.json",
                        help="Path to the JSON search request.")
    parser.add_argument("--output", default="output.json",
                        help="Path to write the JSON response.")
    # tolerate (and ignore) any extra flags the harness may append
    args, _unknown = parser.parse_known_args()

    with open(args.request, "r", encoding="utf-8") as handle:
        request = json.load(handle)

    payload = run_agent(request)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"precursor_agent: wrote {payload['n_results']} result(s) to {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
