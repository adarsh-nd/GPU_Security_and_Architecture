# device_info.md — the ground truth for this machine

All numbers below were reported by the GPU itself via `common/device_query.cu`
(CUDA `cudaGetDeviceProperties`) on 2026-08-28. **These override any spec-sheet
website.** When a later phase needs a hardware number, get it from here.

## The GPU

| Property | Value |
|---|---|
| Name | NVIDIA GeForce RTX 4060 **Laptop** GPU |
| Architecture | Ada Lovelace (AD107) |
| Compute capability | **8.9** |
| SM (multiprocessor) count | 24 |
| Warp size | 32 threads |

## Memory hierarchy — the heart of the project

| Level | Size | Shared or private? |
|---|---|---|
| Shared memory per block | 48 KB | private (per block) |
| Shared memory per SM | 100 KB | private (per SM) |
| **L2 cache** | **32 MB** | **SHARED across all 24 SMs — this is the covert-channel medium** |
| L2 persisting max window | 22 MB | (Ada L2-residency feature; may matter for defense later) |
| Global memory (VRAM) | 8188 MB (~8 GB) | main memory (slow) |
| Memory bus width | 128 bits | — |

### What the L2 = 32 MB number means for the build
- To reliably **flush/fill the shared L2**, a flood buffer must be *bigger than
  32 MB* — plan for streaming through ~64 MB+ to be sure everything resident gets
  evicted. VRAM is 8 GB, so there's plenty of room for big flood buffers.
- The **24 vs 32 MB** ambiguity from spec sites is now resolved: **32 MB**,
  straight from the hardware.

## Handy extras

| Property | Value |
|---|---|
| Max threads per block | 1024 |
| Max threads per SM | 1536 |
| Registers per block | 65536 |
| Core clock (nominal) | 1890 MHz |

## Toolchain / environment

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.3 (WSL2 on Windows) |
| WSL | 2.5.10.0, kernel 6.6.87.2 |
| GPU driver (from Windows) | 615.65.06 (KMD 616.56), CUDA UMD 13.4 |
| CUDA Toolkit (`nvcc`) | 13.3 (V13.3.73) |
| Host compiler (gcc) | 13.3.0 |

## Calibration numbers (fill in as the project progresses)

- **Cache-hit latency (steady state):** ~28 cycles (range 19–29)
- **DRAM-miss latency (steady state):** ~489 cycles (range ~450–774)
- **Cache-hit vs. memory-miss threshold (L1 timer):** **200 cycles** (< 200 = cached, > 200 = from VRAM). ~17× separation, no overlap. Measured 2026-08-30 via timer/latencyV1.cu.
- **L2-channel calibration (M2a, 4 MB probe buffer):** L2-resident ≈ **237 cycles**, L2-evicted (DRAM) ≈ **590 cycles**. **Channel threshold = 400 cycles** (< 400 = resident/bit 0, > 400 = evicted/bit 1). ~2.5× separation. This is the threshold the L2 covert channel uses — higher than the L1 timer's 200 because the channel medium is L2, whose hit latency (~237) sits between L1 (~28) and DRAM (~489). Measured 2026-08-30 via covert/prime_probe.cu.
- **`%globaltimer` granularity:** **~1024 ns (~1 µs)** — the GPU wall-clock advances in 1024 ns steps; a single read costs ~10 ns. Two separate processes read it as one shared timeline (M2-CLOCK confirmed, 2026-08-31). **Constraint:** M2b slot duration must be ≫ 1 µs (plan for tens of µs).
- **Clock drift at steady state (±%):** **~0%** — miss-latency avg was 488.7 cycles on all 20 runs over ~2 min (`common/measure_drift.sh`), SM clock pinned at 1890 MHz during every measurement, temp stable 42–44 °C (no throttling). Measurement bursts are too short to heat the GPU into changing frequency → cycle-counts are reproducible to the decimal. Caveat: a longer sustained load could throttle and shift miss-cycles; not a concern for this project's short bursts. (One 270 MHz reading in the log was nvidia-smi sampling the idle clock *between* runs, not measurement-time drift.)
- Note: first ~32–50 accesses of any cold run are warm-up (TLB/pipeline/cache-fill); discard them before analysis.
- **Saturation memory bandwidth:** **~250 GB/s** (~98% of ~256 GB/s theoretical peak; read-only stream). Measured 2026-09-01 via `arch/bandwidth.cu` / `arch/occupancy.cu`. **This is the ceiling the defense jammer is capped at.**
- **Occupancy knee:** **~256 total warps (~11 warps/SM)** to fully hide DRAM latency (linear MLP scaling up to ~128 warps). Plot: `results/occupancy.png`.
- **DVFS (SM clock) sensor: NOT viable** — SM clock pinned at ~2.65 GHz boost, moves ~0.08% (idle vs. heavy memory neighbor), far below the ~5% needed. A co-tenant's activity does not leak through the clock on this laptop → the defense drops the DVFS sensor and uses temporal (persistence/signature) confirmation instead.
