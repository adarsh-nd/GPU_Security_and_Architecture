#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
const int READS = 8192;
const int THRESHOLD = 400;
const int PROBE_RING_SIZE = 1024 * 1024;
const int ITERS = 200;
using namespace std;

//walks the ring to pull it into L2 (prime step)
__global__ void primeKernel(int* ring, int steps, int* sink) {
    int idx = 0;
    for(int i = 0; i < steps; i++)
    {
        idx = ring[idx];
    }
    sink[0] = idx;
}

//walks the ring timing each read; returns total cycles (probe step)
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

int main() {
    //build the probe ring and upload it
    int* cpuProbeRing = new int[PROBE_RING_SIZE];
    for(int i = 0; i < PROBE_RING_SIZE; i++)
    {
        cpuProbeRing[i] = (i + 32) % PROBE_RING_SIZE;
    }

    int* gpuProbeRing;
    int* gpuSink;
    cudaMalloc((void**)&gpuProbeRing, PROBE_RING_SIZE * sizeof(int));
    cudaMalloc((void**)&gpuSink, sizeof(int));
    cudaMemcpy(gpuProbeRing, cpuProbeRing, PROBE_RING_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    long long* gpuProbeCycles;
    long long cpuProbeCycles;
    cudaMalloc((void**)&gpuProbeCycles, sizeof(long long));

    //200 times: prime, probe, decode present/evicted by threshold and print
    for (int i = 0; i < ITERS; i++) {
        primeKernel<<<1,1>>>(gpuProbeRing, PROBE_RING_SIZE, gpuSink);
        probeKernel<<<1,1>>>(gpuProbeRing, READS, gpuProbeCycles, gpuSink);
        cudaDeviceSynchronize();
        cudaMemcpy(&cpuProbeCycles, gpuProbeCycles, sizeof(long long), cudaMemcpyDeviceToHost);
        double averageCycles = (double) cpuProbeCycles / READS;
        cout << "Iteration #: " << i << ", average: " << averageCycles << ", cycle: "
            << (averageCycles > THRESHOLD ? "EVICTED(1)" : "PRESENT(0)") << endl;
    }

}
