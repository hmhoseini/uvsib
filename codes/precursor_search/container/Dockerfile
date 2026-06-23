# precursor-search agent image for uvsib's SynthesizabilityWorkChain.
#
# Minimal by design: a glibc base + TLS roots + jq + the self-contained Claude
# Code native binary. No Node, Python, git or build toolchain -- this image only
# web-searches literature and summarizes it into JSON. Runs rootless under
# udocker (no daemon, no root). WebSearch/WebFetch are server-side, so the only
# network egress the container needs is HTTPS to the Anthropic API.
#
# Size: ~310 MB uncompressed (the ~234 MB claude binary dominates).

FROM debian:bookworm-slim

# Runtime deps: ca-certificates (TLS to the API) + jq (JSON plumbing).
# curl is used only to fetch the agent binary during build.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates jq curl \
 && rm -rf /var/lib/apt/lists/*

# The entrypoint wrapper (root-owned, world-executable, on PATH).
COPY precursor-agent /usr/local/bin/precursor-agent
RUN chmod 0755 /usr/local/bin/precursor-agent

# Non-root user. udocker maps the runtime uid to the invoking user and owns the
# extracted rootfs, so $HOME stays writable regardless of uid.
ARG AGENT_UID=1000
RUN useradd --create-home --uid ${AGENT_UID} --shell /bin/bash agent
USER agent
ENV HOME=/home/agent
ENV PATH="/home/agent/.local/bin:${PATH}"

# Self-contained Claude Code native binary (bundles its own runtime; no Node).
# install.sh auto-detects glibc and pulls the linux-x64 build.
RUN curl -fsSL https://claude.ai/install.sh | bash

# Lean & offline-friendly: no auto-update, no telemetry/nonessential traffic.
ENV DISABLE_AUTOUPDATER=1 \
    DISABLE_TELEMETRY=1 \
    DISABLE_ERROR_REPORTING=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

# ANTHROPIC_API_KEY (or CLAUDE_CODE_OAUTH_TOKEN), and optionally ANTHROPIC_MODEL
# / ANTHROPIC_BASE_URL, are injected at run time -- never baked into the image.

ENTRYPOINT ["precursor-agent"]
