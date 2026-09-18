#!/usr/bin/env bash
# Run this ON A RENTED H100 HOST (Linux with docker + NVIDIA container toolkit). It traces ONE
# transformer layer of an LLM served by vLLM with the Accel-Sim NVBit tracer, post-processes the
# trace to .tracez, and tars it for the Mac, where ./02_run_sim.sh replays it on the H100 model.
#
# Usage (host):  ./04_h100_trace.sh [-m <hf-model>] [-l <layer>] [-p <prompt>] [-n <max-tokens>]
#                                   [-N <trace-name>] [-w <workdir>] [-L]
#   -m  Hugging Face model id        default Qwen/Qwen2.5-0.5B (ungated, small download)
#   -l  module to trace              default model.layers.12 (a whole decoder block; -L lists names)
#   -p  prompt                       default "The quick brown fox jumps over the lazy dog because"
#   -n  max new tokens               default 8 (prefill + 8 decode steps of that layer get traced)
#   -N  name of the trace set        default <model>__<layer>__<UTC date>
#   -w  work dir on the host         default ~/accelsim-h100
#   -P  patch to git-apply to the framework checkout before the tracer build (e.g.
#       patches/issue561-fix-trywait-drop.patch, the upstream post-processing fix for the split-K deadlock)
#   -L  only print the model's module names and exit (to pick -l)
#   env HF_TOKEN                     passed into the container for gated models
#
# The script re-executes itself inside the container with --inside; you never call that yourself on a
# VM-style host (Lambda, GCP, AWS...). On CONTAINER-style providers (RunPod, Vast.ai) there is no
# docker inside the pod: start the pod from the image below, then run the inner half directly:
#   git clone --branch v2.0.0 --depth 1 https://github.com/accel-sim/accel-sim-framework.git /workspace/asf
#   ACCELSIM_ROOT=/workspace/asf TRACES_ROOT=/workspace/traces bash 04_h100_trace.sh --inside -m ... -l ...
# Time budget on an H100: image pull ~5 min, tracer build ~3 min, model download ~1 min, spinlock
# detection 2 runs + traced run a few minutes. Copy the resulting .tgz to the Mac's ~/accelsim/traces/.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"   # absolute, before any cd

IMAGE="ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8-vllm"   # base image + vllm/transformers
REPO="https://github.com/accel-sim/accel-sim-framework.git"
TAG="v2.0.0"

MODEL="Qwen/Qwen2.5-0.5B"; LAYER="model.layers.12"; PROMPT="The quick brown fox jumps over the lazy dog because"
NTOK=8; NAME=""; WORKDIR="$HOME/accelsim-h100"; LIST=0; INSIDE=0; PATCH=""
while (( $# )); do
  case "$1" in
    -m) MODEL="$2"; shift 2;; -l) LAYER="$2"; shift 2;; -p) PROMPT="$2"; shift 2;; -n) NTOK="$2"; shift 2;;
    -N) NAME="$2"; shift 2;; -w) WORKDIR="$2"; shift 2;; -P) PATCH="$2"; shift 2;; -L) LIST=1; shift;; --inside) INSIDE=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
slug() { echo "$1" | tr '/.' '__' | tr -c 'A-Za-z0-9_\n-' '_'; }
[[ -n "$NAME" ]] || NAME="$(slug "$MODEL")__$(slug "$LAYER")__$(date -u +%Y%m%d)"

# =====================================================================================================
if (( INSIDE == 0 )); then
  echo "== Host checks"
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing: this must run on the GPU host."; exit 1; }
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  command -v docker >/dev/null || { echo "docker missing."; exit 1; }
  docker run --rm --gpus all "$IMAGE" nvidia-smi -L >/dev/null 2>&1 || {
    echo "Pulling $IMAGE (~11 GB) and testing GPU passthrough..."
    docker pull "$IMAGE"
    docker run --rm --gpus all "$IMAGE" nvidia-smi -L || { echo "GPU not visible in container: install nvidia-container-toolkit."; exit 1; }
  }

  if [[ -n "$PATCH" ]]; then
    [[ -f "$PATCH" ]] || { echo "patch not found: $PATCH"; exit 1; }
    PATCH="$(cd "$(dirname "$PATCH")" && pwd)/$(basename "$PATCH")"                    # absolute, before the cd below
  fi

  echo "== Checkout in $WORKDIR"
  mkdir -p "$WORKDIR/traces" "$WORKDIR/hf"
  cd "$WORKDIR"
  [[ -d accel-sim-framework/.git ]] || git clone --branch "$TAG" --depth 1 "$REPO"
  cp "$SELF" accel-sim-framework/04_h100_trace.sh; chmod +x accel-sim-framework/04_h100_trace.sh
  if [[ -n "$PATCH" ]]; then cp "$PATCH" accel-sim-framework/04_upstream.patch; PATCH=/accel-sim/04_upstream.patch; fi

  LISTFLAG=(); (( LIST )) && LISTFLAG=(-L)
  TTY=(); [[ -t 0 ]] && TTY=(-it)          # -it only when we actually have a terminal (nohup/ssh -n runs have none)
  echo "== Entering container"
  exec docker run --rm "${TTY[@]}" --gpus all --ipc=host --ulimit memlock=-1 \
    -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HUB_ENABLE_HF_TRANSFER=0 \
    -v "$WORKDIR/accel-sim-framework:/accel-sim" -v "$WORKDIR/traces:/traces" -v "$WORKDIR/hf:/root/.cache/huggingface" \
    -w /accel-sim "$IMAGE" /bin/bash ./04_h100_trace.sh --inside -m "$MODEL" -l "$LAYER" -p "$PROMPT" -n "$NTOK" -N "$NAME" -w "$WORKDIR" -P "$PATCH" "${LISTFLAG[@]}"
fi

# ================================= inside the container ==============================================
ROOT="${ACCELSIM_ROOT:-/accel-sim}"; TROOT="${TRACES_ROOT:-/traces}"
cd "$ROOT"; mkdir -p "$TROOT"
TR=./util/tracer_nvbit
HOOK=$TR/others/torch_hook
export CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH:-/usr/local/cuda}; export PATH="$CUDA_INSTALL_PATH/bin:$PATH"

echo "== 1. Tracer build deps"
missing=(); command -v bc >/dev/null || missing+=(bc); [[ -f /usr/include/zstd.h ]] || missing+=(libzstd-dev)
if (( ${#missing[@]} )); then
  apt-get update -qq --allow-insecure-repositories 2>&1 | grep -vE '^W:' || true   # unsigned NVIDIA devtools repo in the image
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
fi
python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "== 2. NVBit tracer (tracer_tool.so, post-traces-processing, spinlock_tool.so)"
[[ -d $TR/nvbit_release/core ]] || $TR/install_nvbit.sh
if [[ -n "$PATCH" ]]; then
  # --ignore-whitespace: the framework sources have CRLF line endings
  if git apply --ignore-whitespace --reverse --check "$PATCH" 2>/dev/null; then echo "patch already applied: $PATCH"
  else git apply --ignore-whitespace "$PATCH"; echo "applied $PATCH"; rm -f $TR/tracer_tool/traces-processing/post-traces-processing; fi
fi
if [[ ! -f $TR/tracer_tool/tracer_tool.so || ! -x $TR/tracer_tool/traces-processing/post-traces-processing || ! -f $TR/others/spinlock_tool/spinlock_tool.so ]]; then
  make -C $TR -j"$(nproc)" 2>&1 | grep -vE 'nvcc warning|Entering|Leaving' || true
fi
ls -l $TR/tracer_tool/tracer_tool.so $TR/tracer_tool/traces-processing/post-traces-processing $TR/others/spinlock_tool/spinlock_tool.so

echo "== 3. Trace driver"
cat > $HOOK/llm_trace.py <<'EOF'
"""vLLM + Accel-Sim torch_hook driver: trace the forward pass of one named module."""
import argparse, logging, sys, torch
from vllm import LLM, SamplingParams
from torch_hook import TorchModelHookWrapper, hook_nvbit_to_layer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--layer", action="append", default=[])
ap.add_argument("--prompt", default="Hello"); ap.add_argument("--max-tokens", type=int, default=8)
ap.add_argument("--list-layers", action="store_true"); ap.add_argument("--max-model-len", type=int, default=2048)
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)

# enforce_eager: no CUDA graphs (required so per-kernel hooks fire). Single process (VLLM_ENABLE_V1_MULTIPROCESSING=0).
llm = LLM(model=a.model, enforce_eager=True, max_model_len=a.max_model_len, gpu_memory_utilization=0.6)

def list_layers(model: torch.nn.Module):
    for name, m in model.named_modules():
        print(f"  {name}: {m.__class__.__name__}")

def attach(model: torch.nn.Module, layers):
    w = TorchModelHookWrapper(model)
    for l in layers:
        hook_nvbit_to_layer(w, l)      # NVBit on only inside this module's forward

if a.list_layers:
    llm.collective_rpc(lambda self: list_layers(self.model_runner.model)); sys.exit(0)
if not a.layer:
    sys.exit("pass --layer <module name> (use --list-layers to see them)")
llm.collective_rpc(lambda self: attach(self.model_runner.model, a.layer))
out = llm.generate([a.prompt], SamplingParams(temperature=0.0, max_tokens=a.max_tokens))
print("PROMPT:", repr(a.prompt)); print("OUTPUT:", repr(out[0].outputs[0].text))
EOF

# Environment the upstream run.sh wrapper sets, made explicit so the same driver can run under the
# spinlock tool (two passes) and then the tracer (one pass).
export PYTHONPATH="$PWD/$TR/others:$PWD/$TR/tracer_tool:${PYTHONPATH:-}"
export NVBIT_INSTRUMENTATION_ENABLED=0 ALLOW_REG_VAL_TRACING=1 SPINLOCK_HANDLING_MODE=2
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
DRIVER=(python3 $HOOK/llm_trace.py --model "$MODEL" --prompt "$PROMPT" --max-tokens "$NTOK")

if (( LIST )); then
  echo "== Module names of $MODEL"; "${DRIVER[@]}" --list-layers 2>/dev/null | grep -E '^\s' || true; exit 0
fi

# Layout matches util/job_launching/apps/define-llm-apps.yml (suite llm_layer, exec llm_layer, name layer):
#   /traces/<NAME>/llm_layer/layer/traces/kernelslist.g
OUT="$TROOT/$NAME/llm_layer/layer"
mkdir -p "$OUT/traces" "$OUT/spinlock_detection"
export TRACES_FOLDER="$OUT"

echo "== 4. Spinlock detection (2 passes, finds mbarrier/try_wait spin loops to mark_region in the trace)"
rm -f "$OUT"/spinlock_detection/*
for ph in 0 1; do
  SPINLOCK_PHASE=$ph CUDA_INJECTION64_PATH="$PWD/$TR/others/spinlock_tool/spinlock_tool.so" "${DRIVER[@]}" --layer "$LAYER" > "$OUT/spinlock_phase$ph.log" 2>&1 \
    || { tail -30 "$OUT/spinlock_phase$ph.log"; echo "spinlock phase $ph failed"; exit 1; }
done
ls -l "$OUT/spinlock_detection/" ; wc -l "$OUT/spinlock_detection/spinlock_instructions.txt" 2>/dev/null || true

echo "== 5. Traced run of $LAYER"
rm -f "$OUT"/traces/*
CUDA_INJECTION64_PATH="$PWD/$TR/tracer_tool/tracer_tool.so" "${DRIVER[@]}" --layer "$LAYER" > "$OUT/trace_run.log" 2>&1 || echo "(driver exited non-zero; see $OUT/trace_run.log)"
grep -E 'PROMPT:|OUTPUT:|Writing results|Found nvbit|Error|error' "$OUT/trace_run.log" | head -40 || true
raw=$(ls "$OUT"/traces/kernel-*.trace* 2>/dev/null | wc -l); echo "raw kernel traces: $raw"
(( raw > 0 )) || { echo "No kernels were traced. Check the layer name (-L) and $OUT/trace_run.log"; exit 1; }

echo "== 6. Post-process to .tracez"
$TR/tracer_tool/traces-processing/post-traces-processing "$OUT/traces" -j "$(nproc)" 2>&1 | tee "$OUT/post_processing.log"
echo "Dropped-TRYWAIT lines: $(grep -c 'Dropped .* TRYWAIT' "$OUT/post_processing.log" || true)"
# Keep the raw traces (own archive, not in the replay set): post-processing can only be redone from them.
RAW="$TROOT/$NAME.raw"; rm -rf "$RAW"; mkdir -p "$RAW"
find "$OUT/traces" -maxdepth 1 \( -name '*.trace' -o -name '*.trace.xz' -o -name 'kernelslist' -o -name 'kernelslist_ctx_*' \) -exec cp -p {} "$RAW/" \;
rm -f "$OUT"/traces/*.trace "$OUT"/traces/*.trace.xz "$OUT"/traces/kernelslist
ls "$OUT/traces" | head; echo "kernels in kernelslist.g: $(grep -c . "$OUT/traces/kernelslist.g")"

cat > "$TROOT/$NAME/trace_info.txt" <<EOF
name=$NAME
model=$MODEL
layer=$LAYER
prompt=$PROMPT
max_tokens=$NTOK
date_utc=$(date -u +%FT%TZ)
gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)
vllm=$(python3 -c 'import vllm;print(vllm.__version__)')
accel_sim=$TAG nvbit=1.8
EOF

echo "== 7. Package"
tar -czf "$TROOT/$NAME.tgz" -C "$TROOT" "$NAME"
tar -cf "$TROOT/$NAME.raw.tar" -C "$TROOT" "$NAME.raw"          # already xz-compressed per kernel
du -sh "$TROOT/$NAME" "$TROOT/$NAME.tgz" "$TROOT/$NAME.raw.tar"
cat <<EOF

Done. Archive: $TROOT/$NAME.tgz in the container (= $WORKDIR/traces/$NAME.tgz on a VM host).
Raw (pre-post-processing) traces: $TROOT/$NAME.raw.tar -- copy these too, they cannot be regenerated without the GPU.
Copy it to the Mac and simulate:
  scp <host>:<path>/$NAME.tgz ~/accelsim/traces/ && tar -xzf ~/accelsim/traces/$NAME.tgz -C ~/accelsim/traces/
  (in the accelsim container)  ./02_run_sim.sh llm_layer /traces/$NAME H100-SASS-Accelwattch_SASS_SIM $NAME
EOF
