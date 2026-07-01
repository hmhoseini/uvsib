#!/usr/bin/env bash
# Build the image, export it as a tarball, and import it into udocker.
# Uses podman if present, else docker. Run from this directory.
set -euo pipefail

IMAGE="${IMAGE:-precursor-agent:1.0}"
TAR="${TAR:-precursor-agent-1.0.tar}"
UDOCKER_NAME="${UDOCKER_NAME:-precursor-agent}"

if command -v podman >/dev/null 2>&1; then ENGINE=podman
elif command -v docker >/dev/null 2>&1; then ENGINE=docker
else echo "need podman or docker to build" >&2; exit 1; fi

echo ">> building $IMAGE with $ENGINE"
"$ENGINE" build -t "$IMAGE" .

echo ">> exporting -> $TAR"
"$ENGINE" save -o "$TAR" "$IMAGE"
echo ">> image size:"; du -h "$TAR"

if command -v udocker >/dev/null 2>&1; then
    echo ">> importing into udocker as '$UDOCKER_NAME'"
    udocker load -i "$TAR"
    udocker create --name="$UDOCKER_NAME" "$IMAGE" || true
    # Fakechroot avoids PRoot ptrace overhead; PRoot (default) also works.
    udocker setup --execmode=F3 "$UDOCKER_NAME" || true
    echo ">> done. test with:"
    echo "   ANTHROPIC_API_KEY=... udocker run -v \$PWD:\$PWD -w \$PWD $UDOCKER_NAME \\"
    echo "        --request=examples/request.json --output=output.json"
else
    echo ">> udocker not found here; copy $TAR to the remote and run:"
    echo "   udocker load -i $TAR && udocker create --name=$UDOCKER_NAME $IMAGE"
fi
