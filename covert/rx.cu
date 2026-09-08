#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <time.h>

const int READS = 8192;                    //timed reads per probe (one sample)
const int THRESHOLD = 400;                 //above this many cycles = "evicted"
const int PROBE_RING_SIZE = 1024 * 1024;   //probe ring = 4 MB (lives in the shared L2)
const long long SLOT_NS = 150000000;       //150 ms per slot (must match tx.cu)
const long long SKIP_NS = 50000000;        //ignore the first 50 ms of each slot (guard band)
const int NUM_SLOTS = 48;

using namespace std;

//shared CLOCK_MONOTONIC clock in ns; same timeline as tx
unsigned long long cpuCurrentTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

//walks the ring to load our own data into L2 (prime)
__global__ void primeKernel(int* ring, int steps, int* sink) {
    int idx = 0;
    for (int i = 0; i < steps; i++) idx = ring[idx];
    sink[0] = idx;
}

//times reads of our ring; a big total means our data got evicted (probe)
__global__ void probeKernel(int* ring, int reads, long long* cycles, int* sink) {
    int idx = 0;
    long long totalCycles = 0;
    for (int i = 0; i < reads; i++) {
        long long startCycle = clock64();
        idx = ring[idx];
        totalCycles += clock64() - startCycle;
    }
    cycles[0] = totalCycles;
    sink[0] = idx;
}

int main(int argc, char** argv) {
    //build the probe ring and upload it
    int* cpuProbeRing = new int[PROBE_RING_SIZE];
    for (int i = 0; i < PROBE_RING_SIZE; i++) cpuProbeRing[i] = (i + 32) % PROBE_RING_SIZE;

    int* gpuProbeRing;
    int* gpuSink;
    cudaMalloc((void**)&gpuProbeRing, PROBE_RING_SIZE * sizeof(int));
    cudaMalloc((void**)&gpuSink, sizeof(int));
    cudaMemcpy(gpuProbeRing, cpuProbeRing, PROBE_RING_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    long long* gpuProbeCycles;
    long long cpuProbeCycles;
    cudaMalloc((void**)&gpuProbeCycles, sizeof(long long));

    //shared start time (the number tx printed), from the command line
    unsigned long long startTime = atoll(argv[1]);

    //for each slot: wait, skip the guard band, sample repeatedly, majority-vote the bit
    int decodedBits[NUM_SLOTS];
    for (int k = 0; k < NUM_SLOTS; k++) {
        unsigned long long slotStart = startTime + (unsigned long long)k       * SLOT_NS;
        unsigned long long slotEnd   = startTime + (unsigned long long)(k + 1) * SLOT_NS;

        while (cpuCurrentTime() < slotStart) {}

        //skip the guard band so the previous slot's flood tail can't corrupt this vote
        while (cpuCurrentTime() < slotStart + SKIP_NS) {}

        //prime + probe repeatedly for the rest of the slot, counting evicted samples
        int samples = 0, evicted = 0;
        while (cpuCurrentTime() < slotEnd) {
            primeKernel<<<1,1>>>(gpuProbeRing, READS, gpuSink);
            probeKernel<<<1,1>>>(gpuProbeRing, READS, gpuProbeCycles, gpuSink);
            cudaDeviceSynchronize();
            cudaMemcpy(&cpuProbeCycles, gpuProbeCycles, sizeof(long long), cudaMemcpyDeviceToHost);
            double averageCycles = (double)cpuProbeCycles / READS;
            samples++;
            if (averageCycles > THRESHOLD) evicted++;
        }

        //majority vote: mostly-evicted = 1, mostly-resident = 0 (guard against 0 samples)
        decodedBits[k] = (samples > 0 && evicted * 2 > samples) ? 1 : 0;
        cout << "Slot " << k << ": " << evicted << "/" << samples
             << " evicted -> " << decodedBits[k] << endl;
    }

    //pack the decoded bits back into bytes (MSB-first, matching tx) and print the message
    cout << "decoded message: ";
    for (int c = 0; c < 6; c++) {
        char ch = 0;
        for (int j = 0; j < 8; j++)
            ch = (ch << 1) | decodedBits[c*8 + j];
        cout << ch;
    }
    cout << endl;

    cudaFree(gpuProbeRing);
    cudaFree(gpuSink);
    cudaFree(gpuProbeCycles);
    delete[] cpuProbeRing;
    return 0;
}
