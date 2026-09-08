//BER-test receiver: regenerates the same known bits, decodes the channel, reports BER + throughput. usage: ./rx_ber <startTime> <slot_ns> <nbits>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <cstdlib>
const int READS = 8192;
const int THRESHOLD = 400;
const int PROBE_RING_SIZE = 1024 * 1024;
const unsigned SEED = 1234;         //must match tx_ber

using namespace std;

//shared CLOCK_MONOTONIC clock in ns
unsigned long long cpuCurrentTime() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
//walks the ring to load our data into L2 (prime)
__global__ void primeKernel(int* ring, int steps, int* sink) {
    int idx = 0; for (int i = 0; i < steps; i++) idx = ring[idx]; sink[0] = idx;
}
//times reads of our ring; a big total means our data got evicted (probe)
__global__ void probeKernel(int* ring, int reads, long long* cycles, int* sink) {
    int idx = 0; long long totalCycles = 0;
    for (int i = 0; i < reads; i++) { long long startCycle = clock64(); idx = ring[idx]; totalCycles += clock64() - startCycle; }
    cycles[0] = totalCycles; sink[0] = idx;
}

int main(int argc, char** argv) {
    //parse args; guard band scales with slot size
    unsigned long long startTime = atoll(argv[1]);
    long long SLOT_NS = atoll(argv[2]);
    int NBITS = atoi(argv[3]);
    long long SKIP_NS = SLOT_NS / 3;

    //build the probe ring and upload it
    int* cpuProbeRing = new int[PROBE_RING_SIZE];
    for (int i = 0; i < PROBE_RING_SIZE; i++) cpuProbeRing[i] = (i + 32) % PROBE_RING_SIZE;
    int *gpuProbeRing, *gpuSink;
    cudaMalloc((void**)&gpuProbeRing, PROBE_RING_SIZE * sizeof(int));
    cudaMalloc((void**)&gpuSink, sizeof(int));
    cudaMemcpy(gpuProbeRing, cpuProbeRing, PROBE_RING_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    long long *gpuProbeCycles, cpuProbeCycles;
    cudaMalloc((void**)&gpuProbeCycles, sizeof(long long));

    //regenerate the known bits from the shared seed, plus a buffer for decoded bits
    srand(SEED);
    int* messageBits = new int[NBITS];
    for (int i = 0; i < NBITS; i++) messageBits[i] = rand() & 1;
    int* decodedBits = new int[NBITS];

    //decode each slot: wait, skip guard band, sample repeatedly, majority-vote the bit
    for (int k = 0; k < NBITS; k++) {
        unsigned long long slotStart = startTime + (unsigned long long)k       * SLOT_NS;
        unsigned long long slotEnd   = startTime + (unsigned long long)(k + 1) * SLOT_NS;
        while (cpuCurrentTime() < slotStart) {}
        while (cpuCurrentTime() < slotStart + SKIP_NS) {}
        int samples = 0, evicted = 0;
        while (cpuCurrentTime() < slotEnd) {
            primeKernel<<<1,1>>>(gpuProbeRing, READS, gpuSink);
            probeKernel<<<1,1>>>(gpuProbeRing, READS, gpuProbeCycles, gpuSink);
            cudaDeviceSynchronize();
            cudaMemcpy(&cpuProbeCycles, gpuProbeCycles, sizeof(long long), cudaMemcpyDeviceToHost);
            double averageCycles = (double)cpuProbeCycles / READS; samples++;
            if (averageCycles > THRESHOLD) evicted++;
        }
        decodedBits[k] = (samples > 0 && evicted * 2 > samples) ? 1 : 0;
    }

    //count errors, compute BER and throughput, print
    int errors = 0;
    for (int i = 0; i < NBITS; i++) if (decodedBits[i] != messageBits[i]) errors++;
    double ber = 100.0 * errors / NBITS;
    double throughput = 1e9 / (double)SLOT_NS;       //bits per second
    cout << "SLOT_MS=" << (SLOT_NS / 1000000.0)
         << " THROUGHPUT_BPS=" << throughput
         << " BER=" << ber
         << " ERRORS=" << errors << "/" << NBITS << endl;

    cudaFree(gpuProbeRing); cudaFree(gpuSink); cudaFree(gpuProbeCycles);
    delete[] cpuProbeRing; delete[] messageBits; delete[] decodedBits;
    return 0;
}
