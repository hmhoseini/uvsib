#!/usr/bin/env bash
# Offline test of the wrapper's JSON contract (no network, no API key).
# Stubs `claude` with a canned --output-format json envelope and checks that
# precursor-agent produces a schema-valid output.json that the AiiDA parser
# would accept (top-level dict with a results list, plus formula/n_results/agent).
set -euo pipefail
cd "$(dirname "$0")"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp examples/request.json "$WORK/request.json"

pass=0; fail=0
check() { if eval "$2"; then echo "  ok: $1"; pass=$((pass+1)); else echo "  FAIL: $1"; fail=$((fail+1)); fi; }

run_with_stub() {  # $1 = inner JSON text the fake model "returns"
    local inner="$1"
    cat > "$WORK/claude" <<EOF
#!/usr/bin/env bash
cat >/dev/null               # consume the prompt on stdin
inner=\$(cat <<'INNER'
$inner
INNER
)
jq -n --arg r "\$inner" '{type:"result",subtype:"success",is_error:false,result:\$r}'
EOF
    chmod +x "$WORK/claude"
    ( cd "$WORK" && PATH="$WORK:$PATH" bash "$OLDPWD/precursor-agent" \
        --request=request.json --output=output.json ) >/dev/null 2>&1
}

echo "[1] well-formed JSON from the model"
run_with_stub '{"formula":"Y2Ru2O7","results":[{"doi":"10.1021/jacs.0c01","title":"T","year":2021,"url":"https://doi.org/10.1021/jacs.0c01","synthesis_routes":[{"method":"solid-state","precursors":["Y2O3","RuO2"],"steps":["mix","calcine 1000C/24h/air"],"conditions":{"temperature_C":1000,"time_h":24,"atmosphere":"air"},"product":"Y2Ru2O7","confidence":0.82,"evidence":"..."}]}]}'
O="$WORK/output.json"
check "valid JSON"               "jq -e . '$O' >/dev/null"
check "results is a list"        "jq -e '.results|type==\"array\"' '$O' >/dev/null"
check "n_results == 1"           "[ \$(jq '.n_results' '$O') = 1 ]"
check "formula preserved"        "[ \$(jq -r '.formula' '$O') = Y2Ru2O7 ]"
check "query carries request"    "jq -e '.query.elements|index(\"Ru\")' '$O' >/dev/null"
check "agent.version set"        "jq -e '.agent.version' '$O' >/dev/null"
check "no spurious error"        "jq -e '.agent|has(\"error\")|not' '$O' >/dev/null"

echo "[2] model wraps JSON in markdown fences"
run_with_stub '```json
{"formula":"Y2Ru2O7","results":[]}
```'
check "fences stripped, valid"   "jq -e '.results|type==\"array\"' '$O' >/dev/null"
check "n_results == 0"           "[ \$(jq '.n_results' '$O') = 0 ]"

echo "[3] model returns prose garbage -> safe empty payload"
run_with_stub 'I could not find anything, sorry.'
check "still valid dict"         "jq -e '.results|type==\"array\"' '$O' >/dev/null"
check "error noted"              "jq -e '.agent.error' '$O' >/dev/null"

echo "[4] missing request -> valid empty payload, exit 0"
( cd "$WORK" && PATH="$WORK:$PATH" bash "$OLDPWD/precursor-agent" \
    --request=nope.json --output=output.json ) >/dev/null 2>&1
check "valid dict on bad input"  "jq -e '.results|type==\"array\"' '$O' >/dev/null"
check "invalid_request error"    "[ \$(jq -r '.agent.error' '$O') = invalid_or_missing_request ]"

echo
echo "passed=$pass failed=$fail"
[ "$fail" -eq 0 ]
