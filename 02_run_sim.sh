#!/usr/bin/env bash
# Run this INSIDE the container. Launches a trace-driven simulation, waits for it, collects stats.
#
# Usage: ./02_run_sim.sh <benchmark> <trace-dir> [config] [run-name]
#   benchmark : suite name from ./util/job_launching/apps/define-*.yml (e.g. rodinia_2.0-ft)
#   trace-dir : dir holding <app>/<args>/traces/kernelslist.g, e.g. /traces/rodinia_2.0-ft/9.1
#   config    : <base>[-<modifier>...] from configs/define-standard-cfgs.yml. Default H100-SASS.
#               Power: append -Accelwattch_SASS_SIM (needs accelwattch_*.xml next to the base
#               gpgpusim.config; shipped for TITANX, RTX2060_S, GV100, QV100, TITANV, and
#               01_build_and_traces.sh copies them into SM90_H100). See the note at the end.
#   run-name  : label for this launch (default <benchmark>-<config>)
#
# Smoke test after 01:  ./02_run_sim.sh rodinia_2.0-ft /traces/rodinia_2.0-ft/9.1 QV100-SASS smoke
# With power:           ./02_run_sim.sh rodinia_2.0-ft /traces/rodinia_2.0-ft/9.1 QV100-SASS-Accelwattch_SASS_SIM smoke-power
set -euo pipefail
cd /accel-sim

BENCH="${1:?benchmark name required}"
TRACES="${2:?trace dir required}"
CFG="${3:-H100-SASS}"
NAME="${4:-${BENCH}-${CFG}}"
JL=./util/job_launching

echo "== 0. Environment"
set +u; source ./gpu-simulator/setup_environment.sh >/dev/null; set -u
BIN=./gpu-simulator/bin/release/accel-sim.out
[[ -x "$BIN" ]] || { echo "$BIN missing: run ./01_build_and_traces.sh first."; exit 1; }
command -v nvcc >/dev/null || export PATH="$CUDA_INSTALL_PATH/bin:$PATH"

echo "== 1. Validate arguments"
TRACES="$(cd "$TRACES" && pwd)"
if ! ls "$TRACES"/*/*/traces/kernelslist.g >/dev/null 2>&1 && ! ls "$TRACES"/*/traces/kernelslist.g >/dev/null 2>&1; then
  echo "No <app>/<args>/traces/kernelslist.g under $TRACES. Point at the <suite>/<cuda-version> dir."; exit 1
fi
# Mirror the launcher's own rules: first token is a base config, the rest are composable extras;
# the benchmark must be a suite key in apps/define-*.yml. The launcher itself just crashes on a typo.
python3 - "$JL" "$BENCH" "$CFG" <<'EOF'
import glob, os, sys, yaml
jl, bench, cfg = sys.argv[1:4]
base, extra = {}, {}
for f in glob.glob(os.path.join(jl, "configs", "define-*.yml")):
    for k, v in (yaml.safe_load(open(f)) or {}).items():
        (base if "base_file" in (v or {}) else extra)[k] = v or {}
suites = set()
for f in glob.glob(os.path.join(jl, "apps", "define-*.yml")):
    suites |= set((yaml.safe_load(open(f)) or {}).keys())
ok = True
if bench not in suites:
    print(f"Unknown benchmark '{bench}'. Known: {' '.join(sorted(suites))}"); ok = False
t = cfg.split("-")
if t[0] not in base:
    print(f"Unknown base config '{t[0]}'. Known: {' '.join(sorted(base))}"); ok = False
else:
    d = os.path.dirname(os.path.expandvars(base[t[0]]["base_file"]))
    print(f"   base config dir: {d}")
    if any("accelwattch" in x.lower() for x in t[1:]) and not glob.glob(os.path.join(d, "accelwattch_*.xml")):
        print(f"'{cfg}' enables AccelWattch but {d} has no accelwattch_*.xml; see the note at the end of this script.")
        ok = False
for x in t[1:]:
    if x not in extra:
        print(f"Unknown modifier '{x}'. Known: {' '.join(sorted(extra))}"); ok = False
sys.exit(0 if ok else 1)
EOF

echo "== 2. Local job template"
# v2.0.0 ships a slurm.sim that stages the run in /tmp and relies on squeue + rsync; without Slurm
# the background loop deletes the temp dir immediately and the stdout file is lost. Upstream
# reverted it the day after the release (dev commit d930ad6d). Apply that revert when running
# through the local process manager (procman), i.e. when neither sbatch nor qsub exists.
if ! command -v sbatch >/dev/null && ! command -v qsub >/dev/null && grep -q 'squeue -j' "$JL/slurm.sim"; then
  [[ -f "$JL/slurm.sim.v2.0.0-orig" ]] || cp "$JL/slurm.sim" "$JL/slurm.sim.v2.0.0-orig"
  cat > "$JL/slurm.sim" <<'EOF'
#! /bin/bash
#SBATCH -J REPLACE_NAME
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=REPLACE_MEM_USAGE
#SBACTH --time=200:00:00,
#SBATCH -p REPLACE_QUEUE_NAME
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL
#SBATCH --output=/tmp/REPLACE_NAME.o%j
#SBATCH --error=/tmp/REPLACE_NAME.e%j

copy_output() {
    mv /tmp/REPLACE_NAME.e$SLURM_JOB_ID ./REPLACE_NAME.e$SLURM_JOB_ID
    mv /tmp/REPLACE_NAME.o$SLURM_JOB_ID ./REPLACE_NAME.o$SLURM_JOB_ID
}

trap copy_output ERR

#citing https://stackoverflow.com/questions/35800082/how-to-trap-err-when-using-set-e-in-bash
#Setting -E alongside -e makes any trap on ERR inherited by shell funcs, command substitutions and commands executed in a subshell environment
set -eE

if [ "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" != "1" ]; then
    export GPGPUSIM_ROOT=REPLACE_GPGPUSIM_ROOT
    source $GPGPUSIM_ROOT/setup_environment
else
    echo "Skipping setup_environment - already set"
fi

echo "doing: export -n PTX_SIM_USE_PTX_FILE"
export -n PTX_SIM_USE_PTX_FILE
echo "doing: export LD_LIBRARY_PATH=REPLACE_LIBPATH:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=REPLACE_LIBPATH:$LD_LIBRARY_PATH
echo "doing: cd REPLACE_SUBDIR"
cd REPLACE_SUBDIR
echo "doing: export OPENCL_CURRENT_TEST_PATH=REPLACE_SUBDIR"
export OPENCL_CURRENT_TEST_PATH=REPLACE_SUBDIR
echo "doing: export OPENCL_REMOTE_GPU_HOST=REPLACE_REMOTE_HOST"
export OPENCL_REMOTE_GPU_HOST=REPLACE_REMOTE_HOST
echo "doing REPLACE_BENCHMARK_SPECIFIC_COMMAND"
REPLACE_BENCHMARK_SPECIFIC_COMMAND
echo "doing: export PATH=REPLACE_PATH"
export PATH=REPLACE_PATH

# Uncomment to force blocking torque launches
# this needs to be commented for concurrent kernel ptx mode
echo "doing export CUDA_LAUNCH_BLOCKING=1"
export CUDA_LAUNCH_BLOCKING=1


echo "doing: REPLACE_EXEC_NAME REPLACE_COMMAND_LINE"
REPLACE_EXEC_NAME REPLACE_COMMAND_LINE
copy_output
EOF
  echo "Replaced $JL/slurm.sim with the pre-2.0 template (original kept as slurm.sim.v2.0.0-orig)."
else
  echo "OK."
fi

echo "== 3. Concurrency"
# procman ignores the per-job memory hint and launches one job per core. Each trace-driven job
# needs ~4 GB, so derive the job limit from the container's memory limit instead.
lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
[[ "$lim" == "max" || "$lim" -gt $((1<<60)) ]] && lim=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) * 1024 ))
mem_mib=$(( lim / 1048576 ))
CORES=$(( (mem_mib - 1024) / 4096 )); (( CORES < 1 )) && CORES=1; (( CORES > $(nproc) )) && CORES=$(nproc)
echo "Memory limit ${mem_mib} MiB, $(nproc) CPUs -> ${CORES} concurrent job(s)"

echo "== 4. Launch: -B $BENCH -C $CFG -T $TRACES -N $NAME -c $CORES"
LAUNCH_LOG="/results/.launch__${NAME}.log"; mkdir -p /results
"$JL/run_simulations.py" -B "$BENCH" -C "$CFG" -T "$TRACES" -N "$NAME" -c "$CORES" | tee "$LAUNCH_LOG"
# "Job N queued (<app>-<argfolder> <config>)" lines identify exactly which run dirs belong to this launch.
mapfile -t JOBS < <(grep -oE 'queued \(.* ' "$LAUNCH_LOG" | sed -E 's/^queued \(//; s/ $//')

cat <<EOF
== 5. Monitoring (Ctrl-C only detaches; jobs keep running)
   status:  $JL/job_status.py -N $NAME
   resume:  $JL/monitor_func_test.py -v -N $NAME
   kill:    pkill -f accel-sim.out
   run dirs: /accel-sim/sim_run_\$CUDA_VERSION/<app>/<args>/$CFG/  (stdout = *.o<jobid>)
EOF
"$JL/monitor_func_test.py" -v -N "$NAME" -S 30 || echo "(monitor reported failures; collecting what exists)"

echo "== 6. Collecting stats into /results/$NAME"
OUT="/results/$NAME"
mkdir -p "$OUT"
PYTHONWARNINGS=ignore "$JL/get_stats.py" -N "$NAME"        > "$OUT/app_stats.csv"
PYTHONWARNINGS=ignore "$JL/get_stats.py" -N "$NAME" -k     > "$OUT/per_kernel_stats.csv"
PYTHONWARNINGS=ignore "$JL/get_stats.py" -N "$NAME" -k -K  > "$OUT/per_kernel_instance_stats.csv"
head -20 "$OUT/app_stats.csv"

# Keep the raw simulator stdout and any AccelWattch report per app/args, named power__<app>__<args>.log
# Run dir layout: sim_run_<cuda>/<app>/<args>/<config>/
copied=0
for d in ./sim_run_*/*/*/"$CFG"; do
  [[ -d "$d" ]] || continue
  app=$(basename "$(dirname "$(dirname "$d")")"); args=$(basename "$(dirname "$d")")
  # Only dirs from this launch: other runs may share the config name (e.g. an earlier rodinia run).
  printf '%s\n' "${JOBS[@]}" | grep -qxF "${app}-${args}" || continue
  o=$(ls -t "$d"/*.o[0-9]* 2>/dev/null | head -1 || true)
  [[ -n "$o" ]] && cp "$o" "$OUT/stdout__${app}__${args}.txt"
  [[ -f "$d/accelwattch_power_report.log" ]] && { cp "$d/accelwattch_power_report.log" "$OUT/power__${app}__${args}.log"; copied=$((copied+1)); }
done
echo "AccelWattch reports copied: $copied"
ls -l "$OUT"
echo
echo "Then: python3 03_collect.py $OUT"

cat <<'EOF'

NOTE on H100 power. The AccelWattch modifiers reference accelwattch_sass_sim.xml by bare filename and the
launcher copies *.xml from the base config's directory into the run dir. Upstream ships those XMLs for
TITANX, RTX2060_S, GV100, QV100 and TITANV, and all five are byte-identical V100 calibrations; the
simulator injects SM count and clock at runtime. 01_build_and_traces.sh copies the same files into
SM90_H100 (see README.accelwattch.md there), so H100-SASS-Accelwattch_SASS_SIM runs. Treat H100 watts as
a RELATIVE model (kernel vs kernel, config vs config); absolute calibration needs measured H100 power.
EOF
