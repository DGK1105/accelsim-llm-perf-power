#!/usr/bin/env bash
# Run this INSIDE the container (cwd = /accel-sim). Builds Accel-Sim 2.0 and fetches a small
# published trace set to prove the toolchain works.
#
# Usage: ./01_build_and_traces.sh
#   env GPGPUSIM_BRANCH=<ref>  override the pinned GPGPU-Sim commit (default below)
#   env FORCE_BUILD=1          rebuild even if accel-sim.out already exists
#
# Safe to re-run: pip and cmake are incremental, the trace download is skipped if present.
set -euo pipefail
cd /accel-sim

# GPGPU-Sim (performance model + AccelWattch) is cloned by setup_environment.sh. Upstream
# defaults to the moving "dev" branch; pin the dev commit that shipped with v2.0.0 (2026-08-25)
# so the build is reproducible.
export GPGPUSIM_BRANCH="${GPGPUSIM_BRANCH:-e10018b67a4b668e7b43f89280cf67624f1df4ff}"
# One compile job per ~1 GB of container memory, capped at CPU count (12 jobs in 6 GB gets OOM-killed).
lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
[[ "$lim" == "max" || "$lim" -gt $((1<<60)) ]] && lim=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) * 1024 ))
JOBS=$(( lim / 1073741824 )); (( JOBS < 1 )) && JOBS=1; (( JOBS > $(nproc) )) && JOBS=$(nproc)

echo "== 1. Build dependencies the image may lack"
# The Dockerfile installs cmake but not libzstd-dev (needed by the .tracez trace parser) or
# python3-dev (needed by the pybind11 module CMake always builds). Install only if missing.
missing=()
[[ -f /usr/include/zstd.h ]] || missing+=(libzstd-dev)
python3 -c 'import sysconfig,os,sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)' \
  || missing+=(python3-dev)
command -v cmake >/dev/null || missing+=(cmake)
if (( ${#missing[@]} )); then
  echo "Installing: ${missing[*]}"
  # The image adds NVIDIA's devtools apt source without a valid signature; plain apt-get update
  # exits 100 on it. The Dockerfile itself uses --allow-insecure-repositories, so do the same.
  apt-get update -qq --allow-insecure-repositories 2>&1 | grep -vE '^W:' || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
else
  echo "All present."
fi

echo "== 2. Python deps (into the image's /venv)"
pip3 install -q -r requirements.txt

echo "== 3. Simulator environment (clones GPGPU-Sim @ ${GPGPUSIM_BRANCH:0:12} on first run)"
# setup_environment.sh reads several unset variables, which 'set -u' would turn into a fatal
# error, so relax it just for the source.
set +u
source ./gpu-simulator/setup_environment.sh
set -u
echo "GPGPUSIM_ROOT=$GPGPUSIM_ROOT  ($(git -C "$GPGPUSIM_ROOT" rev-parse --short HEAD))"

echo "== 4. Build Accel-Sim 2.0 with cmake (~10-20 min under Rosetta, -j$JOBS)"
BIN=./gpu-simulator/bin/release/accel-sim.out
if [[ -x "$BIN" && -z "${FORCE_BUILD:-}" ]]; then
  echo "$BIN already built; set FORCE_BUILD=1 to rebuild. Running an incremental build anyway."
fi
# Same invocation as upstream CI: build dir carries the config name so PYTHONPATH from
# setup_environment.sh (build/release) finds the python module.
# IPO off: pybind11 enables -flto for the python module and GCC's lto1 crashes with an internal
# compiler error under Rosetta. The main accel-sim.out never used LTO, so nothing is lost.
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build/release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
cmake --build ./gpu-simulator/build/release -j"$JOBS"
cmake --install ./gpu-simulator/build/release
ls -l "$BIN"
echo "version: $(strings "$BIN" | grep -m1 -oE 'accelsim-commit[^"[:space:]]*' || echo unknown)"

echo "== 5. Simulation configs (base + modifiers, joined with '-', e.g. H100-SASS)"
CFG=./util/job_launching/configs/define-standard-cfgs.yml
echo "Base GPU configs:"
awk '/^[A-Za-z0-9_-]+:/{name=$1} /base_file:/{sub(":","",name); printf "  %s", name} END{print ""}' "$CFG"
echo "AccelWattch modifiers:"
grep -oE '^[A-Za-z0-9_-]*Accelwattch[A-Za-z0-9_-]*' "$CFG" | sed 's/^/  /' | tr '\n' ' '; echo
echo "AccelWattch XML files shipped per GPU (power needs one next to gpgpusim.config):"
for d in "$GPGPUSIM_ROOT"/configs/tested-cfgs/*/; do
  n=0; for x in "$d"accelwattch_*.xml; do [[ -f "$x" ]] && n=$((n+1)); done
  (( n > 0 )) && echo "  $(basename "$d"): $n xml" || true
done
# Upstream ships no XML for SM90_H100, but the five it does ship are byte-identical copies of the V100
# calibration (the runtime injects SM count and clock from gpgpusim.config). Follow that precedent so
# H100-SASS-Accelwattch_SASS_SIM works, and leave a provenance note next to the copies.
H100CFG="$GPGPUSIM_ROOT/configs/tested-cfgs/SM90_H100"
if ! ls "$H100CFG"/accelwattch_*.xml >/dev/null 2>&1; then
  cp "$GPGPUSIM_ROOT"/configs/tested-cfgs/SM7_GV100/accelwattch_*.xml "$H100CFG"/
  cat > "$H100CFG/README.accelwattch.md" <<'EOF2'
# AccelWattch XMLs for SM90_H100 (added by 01_build_and_traces.sh)

These six files are byte-for-byte copies of SM7_GV100/accelwattch_*.xml. That is also how upstream
ships them: the GV100, QV100, TITANV, RTX2060_S and TITANX copies are all identical.
AccelWattch overrides number_of_cores, target_core_clockrate and core clock_rate at runtime from
gpgpusim.config, so the H100 SM count (132) and 1980 MHz clock come from the simulator, not from here.

Everything else (dynamic activity factors, constant_power, idle_core_power, static_cat* lane-activation
powers, McPAT structure sizes, tech node) is the published V100 calibration. Results on H100-SASS are a
RELATIVE power model: good for comparing kernels and configs, not for absolute watts (H100 SXM TDP is
700 W vs 300 W for V100). Absolute calibration needs measured H100 power for the AccelWattch QP solver
(util/accelwattch/), i.e. real hardware.
EOF2
  echo "  SM90_H100: none shipped -> copied the 6 GV100 XMLs (V100 calibration, relative model; see README.accelwattch.md there)"
fi

# App definition for LLM layer traces produced by 04_h100_trace.sh. The trace layout is
# <trace-dir>/llm_layer/layer/traces/kernelslist.g ("name:" fixes the argument folder to "layer").
APPYML=./util/job_launching/apps/define-llm-apps.yml
if [[ ! -f "$APPYML" ]]; then
  cat > "$APPYML" <<'EOF2'
# Added by 01_build_and_traces.sh: suite for one traced LLM layer (see 04_h100_trace.sh).
# Simulate with:  ./02_run_sim.sh llm_layer /traces/<trace-name> H100-SASS[-Accelwattch_SASS_SIM] <run-name>
llm_layer:
    exec_dir: "/traces"
    data_dirs: ""
    execs:
        - llm_layer:
            - args: ""
              name: "layer"
              accel-sim-mem: 8G
EOF2
  echo "wrote $APPYML (suite llm_layer)"
fi

echo "== 6. Published traces"
cat <<'EOF'
The public catalogue behind ./get-accel-sim-traces.py holds only Tesla V100 sets from 2020
(rodinia, parboil, polybench, cutlass, deepbench, ubench). There are NO published H100 or LLM
traces; Hopper/LLM traces are produced by the NVBit + PyTorch hook tracer on real hardware
(phase 2). Never run get-accel-sim-traces.py without -a: its default downloads everything (~160 GB).

Fetching the smallest set (rodinia_2.0-ft, 21 MB) as a toolchain smoke test.
EOF
SMOKE_URL="https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz"
SMOKE_DIR=/traces/rodinia_2.0-ft/9.1          # layout: <suite>/<cuda>/<app>/<args>/traces/kernelslist.g
if [[ -d "$SMOKE_DIR" ]] && ls "$SMOKE_DIR"/*/*/traces/kernelslist.g >/dev/null 2>&1; then
  echo "Already present: $SMOKE_DIR ($(ls -d "$SMOKE_DIR"/*/ | wc -l | tr -d " ") apps)"
else
  mkdir -p /traces
  wget -nv -O /traces/rodinia_2.0-ft.tgz "$SMOKE_URL"
  tar -xzf /traces/rodinia_2.0-ft.tgz -C /traces
  rm -f /traces/rodinia_2.0-ft.tgz
  echo "Extracted to $SMOKE_DIR ($(ls -d "$SMOKE_DIR"/*/ | wc -l | tr -d " ") apps)"
fi

cat <<EOF

Build OK. Next, the smoke test (V100 traces on the V100 model; a few minutes each):
   ./02_run_sim.sh rodinia_2.0-ft $SMOKE_DIR QV100-SASS smoke

Other V100 sets, if wanted (sizes are compressed / on disk):
   ./get-accel-sim-traces.py -d /traces -a tesla-v100/ubench        # 82 MB / 2.6 GB
   ./get-accel-sim-traces.py -d /traces -a tesla-v100/rodinia-3.1   # 1.8 GB / 56 GB
   then: tar -xzf /traces/<set>.tgz -C /traces
EOF
