# GPU Cache Security and Architecture

A CUDA hardware-security project built from scratch on an NVIDIA RTX 4060 Laptop (Ada
Lovelace, AD107, compute 8.9; 24 SMs, 32 MB shared L2, 8 GB GDDR6, ~256 GB/s peak;
CUDA 13.3, Ubuntu 24.04 on WSL2). Every number in this document was measured on that
hardware, predicted from first principles and then checked against the silicon. None of
it is estimated.

## Overview

Two programs that are supposed to be isolated from each other, running as separate
processes with no shared memory or files, can still communicate secretly and spy on each
other. They do it purely by watching how they slow each other down while competing for the
GPU's shared L2 cache. This matters because cloud GPUs are often rented as shared hardware,
so your code and a stranger's code can land on the same chip.

On a real RTX 4060, this project demonstrates five things:

1. The leak crosses the isolation boundary. One process floods L2 while another times its
   own memory and watches latency move from roughly 240 to 600 cycles. Only cache state
   crosses between them; there is no shared buffer.
2. Two cooperating programs can send data through it (a covert channel, error-free up to
   about 25 bits per second), and a single program can eavesdrop on an unaware victim
   (a side channel, with 99.6% busy/idle detection).
3. The mechanism is explained by measuring the hardware itself: bandwidth, occupancy, cache
   sectors, and banks. The leak is simply the documented memory system behaving as
   characterized.
4. A defense works. It detects the attacker and jams the channel until the attacker's data
   is unusable, raising the bit error rate from 0% to 54.7%, at a measured cost of about
   6% rather than 99%.
5. Negative results are reported honestly. A clock-based detector does not leak on this
   hardware, and cache pinning protects your own data within a process but is useless
   across processes.

## Components

### 1. Timing instrument
A single-thread pointer chase (`idx = ring[idx]`) timed with `clock64()`. One thread plus
dependent loads defeats the GPU's two latency-hiding tricks, warp switching and
memory-level parallelism, which exposes the true per-access latency. An L1 hit measures
about 28 cycles against a DRAM miss of about 489 cycles, a roughly 17x gap with no overlap.
Clock drift is about 0%.

### 2. Covert channel
Prime and probe on a 4 MB buffer, sized larger than the 128 KB L1 but well under the 32 MB
L2, so it lives in the shared L2 that serves as the covert medium. An L2 hit measures about
237 cycles against about 590 when evicted, with a decision threshold of 400.

Across two independent processes with their own contexts and memory (no MPS), the
receiver's baseline of about 242 cycles rises to about 595 on 198 of 200 samples under a
concurrent flooding sender, which confirms a tenant-isolation breach.

Both processes read the same `%globaltimer` epoch through inline PTX (granularity about
1 microsecond), and the CPU `CLOCK_MONOTONIC` is shared as well. That supports a
time-slotted protocol with per-slot majority voting and guard bands. The channel sent the
message "HOOKEM" (48 bits) error-free, and a throughput-versus-BER sweep showed error-free
operation up to about 25 bits per second, degrading to about 100 bits per second at a
12.5% bit error rate.

Three lessons were hard-won: timing must use the CPU clock, because a GPU spin-wait
serializes the two processes so the flood and probe never overlap; each slot needs a burst
of samples with a majority vote, because a single sample per slot is too fragile; and a
guard band is needed to absorb flood overshoot from the previous slot.

### 3. Microarchitecture characterization
The GPU's memory system was reverse-characterized from timing alone.

- Saturation bandwidth of about 250 GB/s, roughly 98% of the ~256 GB/s peak, measured with
  a grid-stride kernel and CUDA-event timing.
- An occupancy curve that scales linearly from 1 to 128 warps, with a knee around 256 warps
  (about 11 per SM) and a plateau near 250 GB/s. This is Little's Law read straight off the
  hardware.
- A coalescing cliff of about 7.2x, with bandwidth flooring at stride 8, which recovers the
  32-byte memory sector empirically.
- Bank-conflict serialization of about 9x at 32-way conflicts, which confirms the 32-bank
  shared memory (the conflict degree equals gcd(stride, 32)).
- A negative result: the SM clock shifts only about 0.08% under a heavy neighbor, so a
  clock-based detector does not leak here. It was tested, disproven, and dropped.

### 4. Layered defense
The threat model is a co-scheduled attacker running the covert channel. A single guard
process implements a sense, confirm, and respond pipeline, with a separately tested prevent
layer.

- Sense (`defense/sensor.cu`): the receiver's probe reframed as an alarm. Detection is
  97.8% with a 0% false-alarm rate at rest.
- Confirm (`defense/guard.cu`): a persistence gate, a window of five samples that must all
  agree, cuts a naive reactive loop's roughly 6% false alarms down to 0% while keeping
  96.2% detection, at the cost of a five-sample reaction delay.
- Respond (`defense/guard.cu`): a graduated jammer. A key finding is that a naive reactive
  jammer mirrors the attacker, jamming the 1-slots and resting during the 0-slots, which
  re-transmits the signal instead of destroying it. The fix is a latched clamp-down that
  keeps jamming through the quiet slots. After the fix, the attacker's bit error rate goes
  from 0 to 54.7%, which destroys the channel.
- Respond cost and sweet spot: a naive always-on jammer costs a 123% slowdown. An in-process
  cost sweep showed the jammer is DRAM-bandwidth-bound, a cliff rather than a ramp (1.2% at
  2 blocks, 6% at 8, 89% at 128, 99% at 768). A cross-process BER sweep showed the channel
  dies at just 2 blocks, so the original 768-block cap was about 380x more than needed.
  Cutting it to 8 blocks gives an identical kill at about 6% cost and 0% at rest. The
  headline trade-off is that the attacker's BER goes from 0 to 54.7% for about 6% honest
  cost, and 0% when idle.
- Prevent (`defense/pin_check.cu`, `defense/pin_victim.cu`): L2 pinning through
  `cudaAccessPolicyWindow`. Within a process it works: a pinned ring survives a concurrent
  flood, holding at 213 cycles against 498 unpinned, a 2.3x improvement, with the overlap
  verified in Nsight Systems. Across processes it fails: pinned and unpinned are identical
  (1232 versus 1240 cycles), because the policy is per-context and gives no protection
  against a separate process. This is a clean negative for the realistic deployment threat.

### 5. Presence side channel
`side/victim.cu` does honest GPU work in a 500 ms busy, 500 ms idle pattern, unaware and
with no encoding. `side/spy.cu` runs a prime-and-probe canary and classifies a probe as
busy when its latency exceeds 400 cycles. Accuracy is 99.6% over 11,945 samples (busy
averaging about 1235 cycles against about 246 when idle), with the 0.44% of errors
clustered at the phase transitions. Ground truth is aligned using only the shared monotonic
clock, with no handshake. This is distinct from the covert channel, which has two
cooperating parties, and from the defense, which is disruption: it is espionage against a
target that is not cooperating.

## Isolating the mechanism

Cross-process, on this consumer GPU without MPS, processes time-slice rather than run at
the same time. As a result, any heavy co-tenant, even a compute-only kernel that touches
almost no L2, saturates the channel just as a memory flood does. The cross-process
disruption is therefore timeslice denial, not provably cache eviction.

To isolate the cache mechanism, a single-process rig with two non-blocking streams
(`defense/contention.cu`) forces genuine co-residence on the SMs. Against a matched control
(the same launch geometry and load count, but a 4 KB working set that stays in L1), a
concurrent L2 flood moved a victim probe from 237 to about 595 cycles, a 2.5x change, while
the matched control left it at baseline. This isolates cache contention as a real lever,
separate from raw GPU occupancy. It is the positive result the time-sliced cross-process
setup could not produce, and it explains why. Both results are reported.

A note on method: a two-variable control was caught and fixed, since a compute-only kernel
removed both cache pressure and memory traffic; the same went for a saturated-metric
comparison, where both treatments were pinned at a coin flip, before any claim was trusted.
Stream co-residence was verified on the Nsight Systems timeline. Nsight Compute serializes
kernels and so cannot observe concurrent contention, which is why the live `clock64`
latency is used as the mediator.

## Measured results

| Quantity | Value | Source |
|---|---|---|
| L1-hit latency | ~28 cyc | timer |
| DRAM-miss latency | ~489 cyc | timer |
| L1-hit vs DRAM-miss | ~17x | derived |
| Clock drift | ~0% | timer |
| L2-hit latency | ~237 cyc | `covert/prime_probe.cu` |
| L2-evicted latency | ~590 cyc | `covert/prime_probe.cu` |
| Cross-process resident to evicted | ~242 to ~595 cyc (198/200) | `covert/receiver.cu`, `sender.cu` |
| `%globaltimer` granularity | ~1024 ns | `covert/clock_check.cu` |
| Covert channel, error-free throughput | ~25 b/s (0% BER) | `covert/*_ber.cu` |
| Covert channel, max throughput | ~100 b/s (12.5% BER) | `covert/*_ber.cu` |
| Saturation bandwidth | ~250 GB/s (~98% of peak) | `arch/bandwidth.cu` |
| Occupancy knee | ~256 warps (~11 per SM) | `arch/occupancy.cu` |
| Coalescing cliff | ~7.2x | `arch/coalescing.cu` |
| Memory sector (recovered) | 32 bytes | `arch/coalescing.cu` |
| Bank-conflict penalty | ~9x at 32-way | `arch/bank_conflicts.cu` |
| Shared-memory banks (recovered) | 32 | `arch/bank_conflicts.cu` |
| DVFS shift under load | ~0.08% (negative result) | `arch/dvfs_probe.cu` |
| Defense, sensor detection | 97.8% | `defense/sensor.cu` |
| Defense, false-alarm rate with confirm | 0% (from ~6%) | `defense/guard.cu` |
| Defense, attacker BER under guard | 0% to 54.7% | `defense/guard.cu` |
| Defense, honest cost at 8-block cap | ~6% (0% at rest) | `defense/cost.cu` |
| Defense, naive always-on cost | 123% | `defense/honest_workload.cu` |
| Defense, jammer cap reduction | 768 to 8 blocks (~380x) | `defense/pure_jammer.cu` |
| Isolated L2-contention lever | 2.5x (237 to ~595 cyc) | `defense/contention.cu` |
| L2 pinning, within a process | 2.3x (498 to 213 cyc) | `defense/pin_check.cu` |
| L2 pinning, across processes | none, per-context (1232 vs 1240) | `defense/pin_victim.cu` |
| Side channel, presence detection | 99.6% (busy 1235 vs idle 246 cyc) | `side/spy.cu` |
| Shared L2 cache size | 32 MB | device |

## How it works

- A pointer chase with a single thread measures true latency because it serializes
  dependent loads, defeating memory-level parallelism, and one thread stops the SM from
  hiding the stall by switching to another warp.
- A cache covert channel works because the sender floods the shared L2 and the receiver
  times its own memory. An eviction reads as slow, which encodes a transmitted 1. Only
  cache state crosses the boundary.
- Cross-process is the security-relevant claim: two independent processes have separate
  memory but share one physical L2, so the leak survives the isolation the OS and driver
  provide.
- Latency hiding follows Little's Law: the parallelism needed equals latency times
  throughput. The occupancy curve's linear region and plateau make memory-level parallelism
  visible, and the knee gives the number of warps needed to saturate bandwidth.
- Coalescing and 32-byte sectors: the bandwidth floor at stride 8 (8 floats is 32 bytes)
  pins the sector size empirically.
- Bank conflicts: there are 32 banks, and the worst case at stride 32 serializes all 32
  threads, about a 9x penalty.
- A reactive jammer fails because jamming only when contention is sensed mirrors the
  attacker and re-transmits the signal, leaving the BER at 0. The fix is a latched
  clamp-down that keeps jamming through the quiet slots.
- Isolating cache from occupancy was the central confound. Cross-process, a compute-only
  kernel killed the channel just as a memory flood did, so the disruption was GPU-time
  denial, not cache. Proving cache is a real lever required a single-process, dual-stream
  rig, a matched control, and a de-saturated metric; a concurrent flood then moved the
  victim by 2.5x.
- Right-sizing the response mattered: the jammer's cost is a cliff (bandwidth-bound) while
  the kill is effectively binary (the channel dies at 2 blocks), so cutting the cap by
  about 380x delivers the same kill at roughly 6% cost instead of 99%.

## Skills and tools

Languages and tools: CUDA C/C++, inline PTX (`%globaltimer`), Nsight Systems, Python
(matplotlib), Bash, Linux and WSL2, Git.

GPU and parallel computing: CUDA kernels, the grid/block/thread hierarchy, warps, streaming
multiprocessors, grid-stride loops, occupancy, latency hiding, memory-level parallelism,
memory coalescing, shared-memory bank conflicts, the memory hierarchy (L1, L2, VRAM),
CUDA-event versus `clock64()` timing, CUDA streams and concurrent kernels (non-blocking
streams, SM co-residence), L2 persistence via `cudaAccessPolicyWindow`, and multi-process
co-scheduling.

Computer architecture: cache microarchitecture, L2 cache contention, cache lines and
32-byte sectors, residency and eviction, Little's Law, DVFS behavior, and post-silicon-style
characterization (predict, measure, validate).

Hardware security: covert channels, side-channel analysis, timing attacks, prime and probe,
cache contention channels, tenant isolation and co-tenancy, time-slicing versus concurrency,
a layered defense (detect, confirm, respond, prevent), graduated response and active
jamming, cache-mitigation trade-offs, mechanism isolation, threat modeling, and
BER and channel-capacity analysis.

Engineering practice: microbenchmark design, defeating compiler optimizations (dead-code
elimination sinks, loop-invariant hoisting), reproducible measurement, negative-result
reporting, data visualization, and experimental rigor with ground-truth validation.

## Limitations

- The error-free throughput of about 25 bits per second is limited by a single-thread
  flood; a multi-threaded flood would raise it.
- The ~250 GB/s bandwidth is about 98% of theoretical, which is high because the kernel is
  read-only and partly served from L2. It is a defensible and explainable figure.
- Two-way bank conflicts sit below the loop-overhead noise floor, so the clean signal starts
  at 8-way.
- Without MPS, processes time-slice, so the guard's channel-kill is dominated by GPU-time
  denial. L2 eviction is proven a lever (2.5x) only in the single-process concurrency rig,
  not cleanly separable across processes.
- The 2-block kill is measured against this attacker at a 60 ms slot. A determined attacker
  using error correction, redundancy, or adaptive slotting could resist a minimal jammer,
  which is why the graduated ramp can escalate when a low intensity fails to confirm
  disruption.
- L2 pinning is per-context. It protects within a process (2.3x) but gives no cross-process
  protection, and this is reported as a negative rather than hidden.
- Full secret recovery is out of scope; it would need NVIDIA's undocumented address-to-set
  mapping. This work does presence and activity detection instead.
- This is a CUDA measurement and security project, not an HDL or RTL design.

## Repository layout

| Area | Files | Result |
|---|---|---|
| Timer | `timer/latencyV1.cu` | ~17x hit/miss gap; threshold 200 |
| Covert channel | `covert/prime_probe.cu`, `receiver.cu`, `sender.cu`, `clock_check.cu`, `tx.cu`, `rx.cu`, `tx_ber.cu`, `rx_ber.cu` | cross-process leak; "HOOKEM"; 0 to 12.5% BER sweep |
| Architecture | `arch/bandwidth.cu`, `occupancy.cu`, `coalescing.cu`, `bank_conflicts.cu`, `dvfs_probe.cu` | 250 GB/s; 256-warp knee; 7.2x cliff; 32 banks; DVFS negative |
| Defense | `defense/sensor.cu`, `guard.cu`, `honest_workload.cu`, `cost.cu`, `contention.cu`, `pure_jammer.cu`, `pin_check.cu`, `pin_victim.cu` | 97.8% detection; BER 0 to 54.7% at ~6%; 2.5x lever; pinning per-context |
| Side channel | `side/victim.cu`, `side/spy.cu` | 99.6% presence detection |
| Shared | `common/kernels.cuh` | chase, probe, stream, and flood kernels, plus clocks |

Recorded results and plots (latency histogram, throughput-versus-BER, occupancy,
coalescing, bank-conflict, and defense sweeps) live in `measurements/`.

## Build and run

Each `.cu` file is a standalone program. Compile with `nvcc` for your GPU's architecture
(`sm_89` is Ada, the RTX 4060):

```bash
nvcc -O3 -arch=sm_89 -o defense/guard defense/guard.cu
./defense/guard
```

- The architecture experiments write their CSVs into `measurements/`; create the folder if
  it does not exist.
- The covert channel runs as a pair. Start the receiver with the sender's printed start
  time:
  ```bash
  ./covert/tx_ber <slot_ns> <nbits>                 # prints a start time
  ./covert/rx_ber <start_time> <slot_ns> <nbits>    # decodes and reports BER
  ```
- The presence side channel works the same way: `side/victim` prints its start time for
  `side/spy`.
- Shared kernels live in `common/kernels.cuh`.

## Status

All four stages are complete and the project is feature-complete. Optional follow-ups are an
accuracy-versus-granularity sweep of the side channel and an Nsight Compute L2-hit-rate
corroboration of the contention result.
