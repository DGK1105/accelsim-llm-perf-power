# Filed as https://github.com/accel-sim/accel-sim-framework/issues/561 (2026-09-18)

> **Status (2026-09-18): resolved.** The maintainers traced this to the trace post-processor (`drop_unused_trywaits` reading only the first mbarrier address of multi-lane `ARRIVE` lines), not the simulator's mbarrier model as hypothesised below, and posted a fix in the issue thread. See the README section "The split-K deadlock" and `patches/issue561-fix-trywait-drop.patch`. The text below is the report as originally filed.

Title: v2.0.0: trace-driven H100 sim deadlocks on cuBLAS split-K GEMM (nvjet_sm90_*_splitK) traced from vLLM via torch_hook

## Summary

Replaying one traced transformer layer of Qwen2.5-0.5B (vLLM 0.27.1, H100 SXM, tracer from v2.0.0 with
`SPINLOCK_HANDLING_MODE=2`) under `H100-SASS`, the simulator completes the first 10 kernels and then
reports a deadlock on the 11th, a cuBLAS split-K GEMM. With `-gpgpu_deadlock_detect 0` the run spins
indefinitely at 100% CPU with no further output, so it appears to be a genuine hang rather than a
false detection.

## Environment

- accel-sim-framework v2.0.0 (64653015), gpgpu-sim_distribution dev @ e10018b6, built in the
  `ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8` image (x86_64, under Rosetta on macOS,
  also reproduces the same way as the deadlock message is deterministic)
- Traced on Lambda Cloud 2x H100 80GB SXM5, driver 580.105.08, in
  `ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8-vllm` (vllm 0.27.1, torch 2.13.0+cu130),
  NVBit 1.8, tracer built from v2.0.0 with `make -C util/tracer_nvbit`
- Model `Qwen/Qwen2.5-0.5B`, `enforce_eager=True`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, hook on
  `model.layers.12` via `hook_nvbit_to_layer`, prompt of 11 tokens + 8 decode tokens
- Spinlock detection run first (2 passes with `spinlock_tool.so`, 491 lines in
  `spinlock_detection/spinlock_instructions.txt`), then the traced run with
  `SPINLOCK_HANDLING_MODE=2 ALLOW_REG_VAL_TRACING=1`, then `post-traces-processing -j 52`
  -> 96 `.tracez` kernels, `kernelslist.g` of 96 entries
- Simulated with `run_simulations.py -B <suite> -C H100-SASS-Accelwattch_SASS_SIM -T <dir>` (also
  reproduced with plain `H100-SASS`, and with `accel-sim.out -config gpgpusim.config -trace kernelslist.g`
  on a kernelslist containing only the failing kernel)

## Failing kernel

```
-kernel name = nvjet_sm90_tst_64x8_64x16_4x2_h_bz_splitK_TNT
-grid dim = (8,12,1)
-block dim = (384,1,1)
-shmem = 164292
-nregs = 168
-binary version = 90
```

Kernels 1-10 of the same layer (vllm fused_add_rms_norm, nvjet_sm90 QKV GEMM, rotary_embedding,
reshape_and_cache_flash, cutlass/flash FlashAttention-3 sm90 kernel, at::native fill,
nvjet_sm90 o-proj GEMM, fused_add_rms_norm, nvjet_sm90_tst_128x16_64x11_4x1_v_bz_TNT, act_and_mul)
all simulate to completion (76,502 cycles total).

## Observations

- Decoding the failing kernel with `traceDsm` shows 3921 `REPLAY_START` but only 2769 `REPLAY_END`
  markers; the difference (1152) equals the number of warps (96 CTAs x 12 warps), i.e. every warp's
  last replay region has no `REPLAY_END` before `EXIT`.
- The regions are mbarrier waits:
  ```
  REPLAY_START
  SYNCS.PHASECHK.TRANS64.TRYWAIT R0 R4 ...
  NANOSLEEP.SYNCS 50000
  SYNCS.PHASECHK.TRANS64 R0 R4 ...
  BRA ...
  BRA ...
  REPLAY_END
  ```
- The simulator prints, once per SM, e.g.
  `WARNING: sid 89 warp 8 pc 0x3410 EXIT in replay region - overriding mask 0x0000fcff -> 0xffffffff`
  (from `trace_driven.cc` `get_next_trace_inst`), then after ~100k cycles:
  `GPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core 84 @ gpu_sim_cycle 10432 (+ gpu_tot_sim_cycle ...) (89568 cycles ago)`
  followed by the "shader cores no longer committing instructions" list covering all cores.
- With `-gpgpu_deadlock_detect 0` there is no further output for 25 minutes at 100% CPU.
- `handle_replay_region_exit()` aborts after 100 iterations with a different message; that message is
  never printed, so the warps are not looping in the replay region. They appear to be stalled on the
  mbarrier `TRYWAIT` whose phase never completes in simulation (perhaps because the producer warps
  exited via the overridden-mask `EXIT` path).

- Stripping every `REPLAY_START`/`REPLAY_END` marker from the kernel's trace (decoded with `traceDsm`,
  per-warp `insts =` counts recomputed) and replaying it as plain `.traceg` produces the identical
  deadlock at the same cycle (`last writeback core 21 @ gpu_sim_cycle 10492`), so the replay-region
  logic is not the cause; the warps appear to stall on the `SYNCS.PHASECHK.TRANS64.TRYWAIT` itself.

## Reproduction

The 68 MB trace tarball (`.tracez` files + `kernelslist.g`) and the simulator stdout can be shared on
request; the failing kernel alone is `kernel-2264-ctx_0x35fc7de0.tracez` (6.6 MB).

## Question

Is warp-specialized split-K (`splitK` nvjet kernels, which are what cuBLAS picks for small-M decode
GEMMs in vLLM) expected to work with the v2.0.0 mbarrier / replay-region model, and is there a
recommended tracer setting (`SPINLOCK_HANDLING_MODE`, `SKIP_TMA_MEM`, ...) for these kernels?
