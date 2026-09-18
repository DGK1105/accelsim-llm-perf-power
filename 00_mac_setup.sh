#!/usr/bin/env bash
# Run this ON THE MAC. Installs/starts Docker Desktop if needed, pulls the Accel-Sim 2.0
# image, clones the framework, and drops you into the container.
#
# Usage: ./00_mac_setup.sh [workdir]   (default: ~/accelsim)
#
# Safe to re-run: every step skips work that is already done, and an existing
# "accelsim" container is re-attached instead of re-created.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # resolved before any cd
WORKDIR="${1:-$HOME/accelsim}"
# amd64-only image, ~12 GB compressed / ~25 GB on disk. Tag is case-insensitive on GHCR.
# Its /accel-sim only holds gpu-app-collection (CUDA sample apps), which trace-driven
# runs do not need, so bind-mounting our framework checkout over it is fine.
IMAGE="ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8"
REPO="https://github.com/accel-sim/accel-sim-framework.git"
TAG="v2.0.0"
CONTAINER="accelsim"
MIN_MEM_MIB=8192      # simulator pages traces to ~4 GB/job; 8 GB lets one job + OS fit
MIN_DISK_MIB=102400   # image (~25 GB) + one published trace set (tens of GB)
DOCKER_SETTINGS="$HOME/Library/Group Containers/group.com.docker/settings-store.json"

# Docker Desktop puts its CLI here; make sure a fresh install is visible to this shell.
export PATH="/usr/local/bin:$HOME/.docker/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"

wait_for_docker() {
  local deadline=$((SECONDS + 300))
  until command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      echo "Docker daemon did not come up within 5 min. Open Docker Desktop, finish its first-run"
      echo "dialogs (license, privileged helper), wait for the whale icon to settle, then re-run."
      exit 1
    fi
    sleep 3
  done
}

echo "== 1. Docker Desktop"
if ! command -v docker >/dev/null 2>&1 && [[ ! -d /Applications/Docker.app ]]; then
  if command -v brew >/dev/null 2>&1; then
    read -rp "Docker Desktop is not installed. Install it now with 'brew install --cask docker-desktop'? [Y/n] " ans
    if [[ "${ans:-Y}" =~ ^[Yy]?$ ]]; then
      brew install --cask docker-desktop   # asks for your password once (symlinks CLI tools into /usr/local/bin)
    else
      echo "Install it from https://www.docker.com/products/docker-desktop/ then re-run."; exit 1
    fi
  else
    echo "Docker Desktop not found and Homebrew is missing."
    echo "Install from https://www.docker.com/products/docker-desktop/ then re-run."; exit 1
  fi
fi
if ! (command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1); then
  echo "Starting Docker Desktop (first launch asks you to accept the license; do that in the GUI)..."
  open -a Docker
  wait_for_docker
fi
echo "Docker $(docker version --format '{{.Server.Version}}') is running."

if [[ "$(uname -m)" == "arm64" ]]; then
  echo "== 2. Apple Silicon: Rosetta and Docker Desktop resources"
  if [[ ! -f /Library/Apple/usr/share/rosetta/rosetta ]]; then
    echo "macOS Rosetta 2 is missing; installing (needed for Docker's x86_64 emulation)."
    softwareupdate --install-rosetta --agree-to-license
  fi

  # Read Docker Desktop's settings so we only nag when something is actually wrong.
  check_settings() {
    python3 - "$DOCKER_SETTINGS" "$MIN_MEM_MIB" "$MIN_DISK_MIB" <<'EOF'
import json, sys
path, min_mem, min_disk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    s = json.load(open(path))
except Exception:
    sys.exit(2)                       # unknown layout: fall back to manual confirmation
rosetta = s.get("UseVirtualizationFrameworkRosetta")
mem, disk = s.get("MemoryMiB"), s.get("DiskSizeMiB")
if rosetta is None or mem is None:
    sys.exit(2)
print(f"   Rosetta emulation: {'on' if rosetta else 'OFF'}   Memory: {mem} MiB   Disk limit: {disk} MiB")
bad = 0
if not rosetta:
    print("   -> enable Settings > General > 'Use Rosetta for x86_64/amd64 emulation on Apple Silicon'"); bad = 1
if mem < min_mem:
    print(f"   -> raise Settings > Resources > Memory to at least {min_mem // 1024} GB (12 GB recommended)"); bad = 1
if disk is not None and disk < min_disk:
    print(f"   (warning) Disk limit under {min_disk // 1024} GB; large trace sets may not fit")
sys.exit(bad)
EOF
  }
  rc=0; check_settings || rc=$?
  if (( rc == 2 )); then
    cat <<'EOF'
Could not read Docker Desktop settings automatically. Please confirm in Docker Desktop:
  Settings > General  : "Use Rosetta for x86_64/amd64 emulation on Apple Silicon" is enabled
  Settings > Resources: Memory >= 8 GB (12 GB recommended), Disk >= 100 GB
Press Enter once that is done (Ctrl-C to abort).
EOF
    read -r
  elif (( rc == 1 )); then
    echo "Fix the settings above in Docker Desktop (Apply & restart), then press Enter (Ctrl-C to abort)."
    read -r
    wait_for_docker
    check_settings || { echo "Settings still not satisfied; continuing anyway, but expect slow or OOM-killed runs."; }
  fi
fi

echo "== 3. Image: $IMAGE"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Already present (run 'docker pull --platform linux/amd64 $IMAGE' to refresh)."
else
  echo "Pulling (~12 GB compressed; 10 to 20 min on a fast link)..."
  docker pull --platform linux/amd64 "$IMAGE"
fi

echo "== 4. Accel-Sim $TAG checkout in $WORKDIR"
mkdir -p "$WORKDIR/traces" "$WORKDIR/results"
cd "$WORKDIR"
if [[ ! -d accel-sim-framework/.git ]]; then
  git clone --branch "$TAG" --depth 1 "$REPO"
else
  echo "accel-sim-framework already cloned ($(git -C accel-sim-framework describe --tags --always))."
fi
# The copies of the in-container scripts live in the checkout so they show up at /accel-sim/.
# This project directory is the source of truth; changed files are re-copied.
for f in 01_build_and_traces.sh 02_run_sim.sh 03_collect.py; do
  src="$SCRIPT_DIR/$f"; dst="accel-sim-framework/$f"
  if [[ ! -f "$src" ]]; then echo "   (missing $src, skipping)"; continue; fi
  if [[ ! -f "$dst" ]] || ! cmp -s "$src" "$dst"; then
    cp "$src" "$dst" && chmod +x "$dst" && echo "   copied $f"
  fi
done

echo "== 5. Container: $CONTAINER"
echo "Inside the container, run:  ./01_build_and_traces.sh"
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  if [[ "$(docker container inspect -f '{{.State.Running}}' "$CONTAINER")" == "true" ]]; then
    echo "Container is running; opening a shell in it."
    exec docker exec -it -w /accel-sim "$CONTAINER" /bin/bash
  fi
  echo "Container exists; re-attaching (docker start -ai)."
  exec docker start -ai "$CONTAINER"
fi

# Cap the container at the VM's memory minus 2 GB headroom, but never above 12 GB.
vm_mib=$(( $(docker info --format '{{.MemTotal}}') / 1048576 ))
mem_mib=$(( vm_mib - 2048 )); (( mem_mib > 12288 )) && mem_mib=12288
(( mem_mib < 4096 )) && { echo "Docker VM has only ${vm_mib} MiB; give it more memory first."; exit 1; }
echo "Creating container with --memory ${mem_mib}m (Docker VM has ${vm_mib} MiB)."
exec docker run -it --platform linux/amd64 \
  --name "$CONTAINER" \
  --memory "${mem_mib}m" \
  -v "$WORKDIR/accel-sim-framework:/accel-sim:rw" \
  -v "$WORKDIR/traces:/traces:rw" \
  -v "$WORKDIR/results:/results:rw" \
  -w /accel-sim \
  "$IMAGE" /bin/bash
