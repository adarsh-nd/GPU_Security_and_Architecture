# GPU Cache Security — Cross-Process Covert Channel, Side Channel & Defense

A from-scratch CUDA hardware-security project on an **NVIDIA RTX 4060 Laptop** (Ada
Lovelace, AD107, compute 8.9; 24 SMs, 32 MB shared L2, 8 GB GDDR6, ~256 GB/s peak;
CUDA 13.3, Ubuntu 24.04 / WSL2). Every number below was measured on that hardware —
predicted from first principles, then validated against silicon. Nothing is estimated.

---

## In a nutshell (what this proves)

**Two programs that are supposed to be isolated — separate processes, no shared memory
or files — can secretly communicate, and spy on each other, purely by watching how they
slow each other down while contending for the GPU's shared L2 cache.**

This matters because cloud GPUs are rented *shared*: your code and a stranger's can land
on the same chip. On a real RTX 4060 this project demonstrates:

1. **The leak crosses the isolation boundary** — one process floods L2, another times its
   own memory and sees ~240 → ~600 cycles. Only cache *state* crosses, no shared buffer.
2. **Two spies can send data** through it (covert channel, error-free to ~25 b/s), and
   **one spy can eavesdrop on an unaware victim** (side channel, **99.6%** busy/idle
   detection).
3. **The mechanism is explained by measuring the hardware** (bandwidth, occupancy, cache
   sectors, banks) — the leak is the documented memory system behaving as characterized.
4. **A defense works** — it detects the attacker and jams the channel until the attacker's
   data is garbage (BER 0% → 54.7%), at an honestly-measured ~6% cost (not 99%).
5. **Negative results are reported honestly** — a clock-based detector doesn't leak here,
   and cache "pinning" protects your data in-process but is useless cross-process.

---

## Project descriptions (pick by space)

**Short:** Built a cross-process GPU L2-cache covert channel *and* side channel in CUDA,
empirically characterized the RTX 4060's microarchitecture from timing alone, and built a
measured layered defense.

**Long:** Built a cycle-accurate cache timer (~17× hit/miss separation); proved two
co-scheduled *separate processes* can covertly communicate through the shared L2 cache
(~25 b/s error-free to ~100 b/s at 12.5% BER) and that a spy can detect an *unaware*
victim's activity at 99.6%; reverse-characterized GPU microarchitecture (~250 GB/s
saturation bandwidth, latency-hiding occupancy curve, 7.2× coalescing cliff, 32-way bank
conflicts); and built a layered defense — a cache sensor (97.8% detect, 0% false alarm
after a persistence gate) driving a graduated jammer that collapses the attacker's channel
(BER 0→54.7%) at ~6% honest-workload cost, with the disruption mechanism isolated to a
2.5× L2-contention lever via a dual-stream concurrency rig — reported with honest
security-vs-performance trade-offs and negative results.

---

## What was built (the arc)

### 1. Timing instrument
A single-thread **pointer-chase** (`idx = ring[idx]`) timed with `clock64()` — one thread
+ dependent loads defeat the GPU's two latency-hiding tricks (warp-switching and
memory-level parallelism), exposing true per-access latency. **L1-hit ≈ 28 cyc vs
DRAM-miss ≈ 489 cyc → ~17× gap**, no overlap; clock drift ≈ 0%.

### 2. Covert channel (the headline attack)
- **Prime + Probe** on a 4 MB buffer (> 128 KB L1 but ≪ 32 MB L2 → lives in the *shared*
  L2, the covert medium): **L2-hit ≈ 237 cyc vs evicted ≈ 590 cyc**, threshold 400.
- **Cross-process (no MPS):** two independent processes (own contexts/memory); receiver
  baseline ~242 → **~595 cyc on 198/200 samples** under a concurrent flooding sender.
  **Tenant-isolation breach confirmed.**
- **Shared clock:** both processes read the same `%globaltimer` epoch (inline PTX),
  granularity ~1 µs; the CPU `CLOCK_MONOTONIC` is likewise shared → time-slotted protocol
  with per-slot majority vote and guard bands.
- **Message + capacity:** sent "HOOKEM" (48 bits) error-free; throughput-vs-BER sweep —
  **error-free to ~25 b/s, degrading to ~100 b/s at 12.5% BER.**
- Hard-won lessons: time on the **CPU clock** (a GPU spin-wait serializes the two
  processes so flood and probe never overlap); **per-slot burst + majority vote** (one
  sample/slot is too fragile); **guard band** against flood overshoot.

### 3. GPU architecture characterization (the depth)
Reverse-characterized the microarchitecture from timing alone:
- **~250 GB/s** saturation bandwidth (~98% of ~256 peak; grid-stride, CUDA-event timing).
- **Occupancy / latency hiding:** perfectly linear 1→128 warps, **knee ~256 warps
  (~11/SM)**, plateau ~250 GB/s — Little's Law drawn from silicon.
- **Coalescing:** bandwidth floors at stride 8 → **~7.2× cliff**, empirically recovering
  the **32-byte memory sector**.
- **Bank conflicts:** **~9× serialization at 32-way** → confirms the **32-bank** shared
  memory (conflict degree = gcd(stride, 32)).
- **DVFS negative:** SM clock shifts only ~0.08% under a heavy neighbor → a clock-based
  detector does *not* leak here; tested, disproven, dropped.

### 4. Layered defense (the mitigation-trade-off result)
Threat model: a co-scheduled attacker running the covert channel. A single **guard**
process implements SENSE → CONFIRM → RESPOND, plus a separately-tested PREVENT layer.
- **SENSE** (`defense/sensor.cu`): the receiver's probe reframed as an alarm — **97.8%
  detection, 0% false alarm** at rest.
- **CONFIRM** (`defense/guard.cu`): a persistence gate (window of 5, all must agree) cuts
  a naive reactive loop's ~6% false alarms to **0%** while keeping 96.2% detection; cost
  is a ~5-sample reaction delay.
- **RESPOND** (`defense/guard.cu`): a graduated jammer. **Key finding — a reactive jammer
  *mirrors* the attacker** (jams the `1`-slots, rests the `0`-slots) and doesn't break the
  channel; the fix is a **latched clamp-down** that sustains jamming through the quiet
  slots. After the fix: **attacker BER 0 → 54.7%** (channel destroyed).
- **RESPOND cost / sweet-spot:** the naive always-on jammer cost **123%** slowdown. An
  in-process cost sweep showed the jammer is DRAM-bandwidth-bound — a cliff, not a ramp
  (2 blk=1.2%, 8=6%, 128=89%, 768=99%). But a cross-process BER sweep showed the channel
  **dies at just 2 blocks** (binary: alive=0 / dead=54.7%). So the original 768-block cap
  was **~380× overkill**; cut to **8 blocks → identical kill at ~6% cost, 0% at rest.**
  The money plot: **0 → 54.7% attacker BER for ~6% honest cost, 0% when idle.**
- **PREVENT** (`defense/pin_check.cu`, `pin_victim.cu`): L2 pinning via
  `cudaAccessPolicyWindow`. **In-process it WORKS** (pinned ring survives a concurrent
  flood, 498 → 213 cyc, 2.3×; nsys-verified overlap). **Cross-process it FAILS** (1232 vs
  1240 cyc — identical): the policy is **per-context** and gives zero protection against a
  foreign process. Clean negative for the deployment threat.

### 5. Presence side-channel (the SCA result)
`side/victim.cu` does honest GPU work in a 500 ms busy / 500 ms idle pattern (unaware, no
encoding); `side/spy.cu` runs a prime+probe canary and classifies `latency > 400 → busy`.
**99.6% accuracy** over 11,945 samples (**busy ~1235 cyc vs idle ~246 cyc**); the 0.44%
errors cluster at phase transitions. Ground truth aligned via only the shared monotonic
clock — no handshake. Distinct from the covert channel (two colluders) and the defense
(disruption): this is espionage on a target that isn't cooperating.

---

## Mechanism isolation (why the analysis is honest)

Cross-process, on this consumer GPU (no MPS), processes **time-slice** rather than run
concurrently. So *any* heavy co-tenant — even a compute-only kernel touching ~0 L2 —
saturates the channel identically to a memory flood: cross-process disruption is
**timeslice denial**, not provably cache eviction. To isolate the cache mechanism, a
**single-process, dual-non-blocking-stream** rig (`defense/contention.cu`) gives true SM
co-residence; against a **matched control** (same launch geometry + same load count, but
an L1-resident 4 KB working set), a concurrent L2 flood moved a victim probe **237 → ~595
cyc (2.5×)** while the matched control left it at baseline — **cache contention isolated as
a real lever, separate from GPU occupancy.** This positive is what the time-sliced
cross-process setup could not produce, and it explains why. Both are reported.

*Methodology notes:* caught and fixed a two-variable control (compute-only removed cache
pressure *and* memory traffic) and a saturated-metric comparison (both treatments pinned
at coin-flip) before trusting any claim; verified stream co-residence on the Nsight Systems
timeline; noted Nsight Compute serializes kernels and can't observe concurrent contention,
so the live `clock64` latency is the mediator.

---

## Every measured number

| Quantity | Value | Source |
|---|---|---|
| L1-hit latency | ~28 cyc | timer |
| DRAM-miss latency | ~489 cyc | timer |
| L1-hit vs DRAM-miss | ~17× | — |
| Clock drift | ~0% | timer |
| L2-hit latency | ~237 cyc | `covert/prime_probe.cu` |
| L2-evicted latency | ~590 cyc | `covert/prime_probe.cu` |
| Cross-process resident → evicted | ~242 → ~595 cyc (198/200) | `covert/receiver.cu`/`sender.cu` |
| `%globaltimer` granularity | ~1024 ns | `covert/clock_check.cu` |
| Covert channel: error-free throughput | ~25 b/s (0% BER) | `covert/*_ber.cu` |
| Covert channel: max throughput | ~100 b/s (12.5% BER) | `covert/*_ber.cu` |
| Saturation bandwidth | ~250 GB/s (~98% peak) | `arch/bandwidth.cu` |
| Occupancy knee | ~256 warps (~11/SM) | `arch/occupancy.cu` |
| Coalescing cliff | ~7.2× | `arch/coalescing.cu` |
| Memory sector (recovered) | 32 bytes | `arch/coalescing.cu` |
| Bank-conflict penalty | ~9× at 32-way | `arch/bank_conflicts.cu` |
| Shared-memory banks (recovered) | 32 | `arch/bank_conflicts.cu` |
| DVFS shift under load | ~0.08% (negative) | `arch/dvfs_probe.cu` |
| Defense: sensor detection | 97.8% | `defense/sensor.cu` |
| Defense: false-alarm (with confirm) | 0% (from ~6%) | `defense/guard.cu` |
| Defense: attacker BER under guard | 0% → 54.7% | `defense/guard.cu` |
| Defense: honest cost (8-block cap) | ~6% (0% at rest) | `defense/cost.cu` |
| Defense: naive always-on cost | 123% | `defense/honest_workload.cu` |
| Defense: jammer cap reduction | 768 → 8 blocks (~380×) | `defense/pure_jammer.cu` |
| Mechanism: isolated L2-contention lever | 2.5× (237 → ~595 cyc) | `defense/contention.cu` |
| L2 pinning (in-process) | 2.3× (498 → 213 cyc) | `defense/pin_check.cu` |
| L2 pinning (cross-process) | none — per-context (1232 vs 1240) | `defense/pin_victim.cu` |
| Side channel: presence detection | 99.6% (busy 1235 vs idle 246 cyc) | `side/spy.cu` |
| Shared L2 cache size | 32 MB | device |

---

## Talking points (interview depth)

- **Why a pointer-chase + single thread measures true latency:** serializes dependent
  loads to defeat memory-level parallelism, and one thread stops the SM from hiding the
  stall by switching warps.
- **How a cache covert channel works:** sender floods shared L2, receiver times its *own*
  memory — eviction = slow = a transmitted `1`; only cache state crosses the boundary.
- **Why cross-process is the security claim:** two independent processes have separate
  memory but share one physical L2 — the leak survives OS/driver isolation.
- **Latency hiding / Little's Law:** needed parallelism = latency × throughput; the
  occupancy curve's linear region + plateau make MLP visible; the knee gives the warps to
  saturate bandwidth.
- **Coalescing & 32-byte sectors:** the bandwidth floor at stride 8 (8 floats = 32 B)
  empirically pins the sector size.
- **Bank conflicts:** 32 banks; worst case at stride 32 serializes all 32 threads (~9×).
- **Why a reactive jammer fails:** naively jamming on sensed contention *mirrors* the
  attacker and re-transmits the signal (BER stayed 0); the fix is a latched clamp-down
  that sustains jamming through the quiet slots.
- **Isolating cache from occupancy (the confound):** cross-process, a compute-only kernel
  killed the channel identically to a memory flood → the disruption was GPU-time denial,
  not cache. Proving cache *is* a lever needed a single-process dual-stream rig, a matched
  control, and a de-saturated metric — then a concurrent flood moved the victim 2.5×.
- **Right-sizing the response:** the jammer's cost is a cliff (bandwidth-bound), the kill
  is binary (dies at 2 blocks) → cut the cap ~380× for the same kill at ~6% vs 99% cost.
- **Honest negatives:** the SM clock doesn't leak co-tenant activity (~0.08%), so a DVFS
  detector was dropped; L2 pinning is per-context, so it can't defend cross-process.

---

## Skills & keywords

**Languages / tools:** CUDA C/C++ · inline PTX (`%globaltimer`) · Nsight Systems · Python
(matplotlib) · Bash · Linux / WSL2 · Git.

**GPU / parallel computing:** CUDA kernels · grid/block/thread hierarchy · warps · SMs ·
grid-stride loops · occupancy · latency hiding · memory-level parallelism · memory
coalescing · shared-memory bank conflicts · memory hierarchy (L1/L2/VRAM) · CUDA-event vs
`clock64()` timing · **CUDA streams / concurrent kernels** (non-blocking streams, SM
co-residence) · **L2 persistence / `cudaAccessPolicyWindow`** · multi-process co-scheduling.

**Computer architecture:** cache microarchitecture · L2 cache contention · cache lines &
32-byte sectors · residency/eviction · Little's Law · DVFS behavior · post-silicon-style
characterization (predict → measure → validate).

**Hardware security:** covert channels · **side-channel analysis (SCA)** · timing attacks ·
Prime + Probe · cache contention channels · tenant isolation / co-tenancy · time-slicing vs
concurrency · **layered defense** (detect → confirm → respond → prevent) · **graduated
response / active jamming** · **cache-mitigation trade-offs** · mechanism isolation · threat
modeling · BER / channel-capacity analysis.

**Engineering practice:** microbenchmark design · defeating compiler optimizations (DCE
sinks, loop-invariant hoisting) · reproducible measurement · negative-result reporting ·
data visualization · experimental rigor / ground-truth validation.

---

## Honest caveats (own these — they read as maturity)

- **Throughput floor:** ~25 b/s error-free is limited by a single-thread flood; a
  multi-threaded flood would raise it.
- **Bandwidth near peak:** ~250 GB/s is ~98% of theoretical, high because the kernel is
  read-only and partly L2-served — defensible and explainable.
- **Bank-conflict noise floor:** 2-way conflicts sit below the loop-overhead floor; the
  clean signal starts at 8-way.
- **Cross-process disruption mechanism:** no MPS here means processes time-slice, so the
  guard's channel-kill is dominated by GPU-time denial; L2 eviction is proven a lever
  (2.5×) only in the single-process concurrency rig, not cleanly separable cross-process.
- **Jammer robustness:** the 2-block kill is against this attacker at a 60 ms slot; a
  determined attacker (error-correction, redundancy, adaptive slotting) could resist a
  minimal jammer — the graduated ramp exists to escalate when a low intensity fails.
- **L2 pinning is per-context:** protects in-process (2.3×) but gives no cross-process
  protection — reported as a negative, not hidden.
- **Secret recovery out of scope:** full key-recovery needs NVIDIA's undocumented
  address→cache-set mapping; this work does presence/activity detection instead.
- **No RTL:** this is a CUDA measurement/security project, not an HDL design.

---

## File map

| Area | Files | Result |
|---|---|---|
| Timer | `timer/latencyV1.cu` | ~17× hit/miss; threshold 200 |
| Covert channel | `covert/prime_probe.cu`, `receiver.cu`/`sender.cu`, `clock_check.cu`, `tx.cu`/`rx.cu`, `tx_ber.cu`/`rx_ber.cu` | cross-process leak; "HOOKEM"; 0→12.5% BER sweep |
| Architecture | `arch/bandwidth.cu`, `occupancy.cu`, `coalescing.cu`, `bank_conflicts.cu`, `dvfs_probe.cu` | 250 GB/s; 256-warp knee; 7.2× cliff; 32 banks; DVFS negative |
| Defense | `defense/sensor.cu`, `guard.cu`, `honest_workload.cu`, `cost.cu`, `contention.cu`, `pure_jammer.cu`, `pin_check.cu`, `pin_victim.cu` | 97.8% detect; BER 0→54.7% @ ~6%; 2.5× lever; pinning per-context |
| Side channel | `side/victim.cu`, `side/spy.cu` | 99.6% presence detection |
| Shared | `common/kernels.cuh` | chase / probe / stream / flood kernels + clocks |

Figures live in `results/` (latency histogram, throughput-vs-BER, occupancy, coalescing,
bank-conflict, and defense sweeps).

---

## Build & run

Each `.cu` is a standalone program; compile with `nvcc` for your GPU's arch
(`sm_89` = Ada / RTX 4060):

```bash
nvcc -O3 -arch=sm_89 -o defense/guard defense/guard.cu
./defense/guard
```

- **Arch experiments** write their CSVs into `measurements/` (create the folder if missing).
- **Covert channel** runs as a pair — start the receiver with the sender's printed start time:
  ```bash
  ./covert/tx_ber <slot_ns> <nbits>                 # prints a start time
  ./covert/rx_ber <start_time> <slot_ns> <nbits>    # decodes, reports BER
  ```
- **Presence side-channel** works the same way — `side/victim` prints its start time for `side/spy`.
- Shared kernels live in `common/kernels.cuh`; `measurements/` holds the recorded results and plots.

## Status

**All four stages complete — the project is feature-complete.** Optional polish:
accuracy-vs-granularity sweep of the side channel, and Nsight Compute L2-hit-rate
corroboration of the contention result.
