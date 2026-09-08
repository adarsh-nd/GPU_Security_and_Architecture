#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
const int READS = 8192;
const int FLOOD_STEPS = 1024 * 1024;
const int THRESHOLD = 400;
const int PROBE_RING_SIZE = 1024 * 1024; //fits in L2
const int FLOOD_RING_SIZE = 16 * PROBE_RING_SIZE; //overflows L2

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
    //build the probe ring (fits L2) and a 16x-bigger flood ring (overflows L2), upload both
    int* cpuProbeRing = new int[PROBE_RING_SIZE];
    int* cpuFloodRing = new int[FLOOD_RING_SIZE];
    for(int i = 0; i < PROBE_RING_SIZE; i++)
    {
        cpuProbeRing[i] = (i + 32) % PROBE_RING_SIZE;
    }

    for(int i = 0; i < FLOOD_RING_SIZE; i++)
    {
        cpuFloodRing[i] = (i + 32) % FLOOD_RING_SIZE;
    }
    int* gpuProbeRing;
    int* gpuFloodRing;
    int* gpuSink;
    cudaMalloc((void**)&gpuProbeRing, PROBE_RING_SIZE * sizeof(int));
    cudaMalloc((void**)&gpuFloodRing, FLOOD_RING_SIZE * sizeof(int));
    cudaMalloc((void**)&gpuSink, sizeof(int));
    cudaMemcpy(gpuProbeRing, cpuProbeRing, PROBE_RING_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuFloodRing, cpuFloodRing, FLOOD_RING_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    long long* gpuProbeCycles;
    long long cpuProbeCycles;
    cudaMalloc((void**)&gpuProbeCycles, sizeof(long long));

    //bit 0: prime the probe ring, don't flood, probe -> should stay fast (present)
    primeKernel<<<1,1>>>(gpuProbeRing, PROBE_RING_SIZE, gpuSink);
    probeKernel<<<1,1>>>(gpuProbeRing, READS, gpuProbeCycles, gpuSink);
    cudaDeviceSynchronize();
    cudaMemcpy(&cpuProbeCycles, gpuProbeCycles, sizeof(long long), cudaMemcpyDeviceToHost);
    double quietAverage = (double) cpuProbeCycles/READS;

    //bit 1: prime the probe ring, then flood to evict it, probe -> should be slow (evicted)
    primeKernel<<<1,1>>>(gpuProbeRing, PROBE_RING_SIZE, gpuSink);
    primeKernel<<<1,1>>>(gpuFloodRing, FLOOD_STEPS, gpuSink);
    probeKernel<<<1,1>>>(gpuProbeRing, READS, gpuProbeCycles, gpuSink);
    cudaDeviceSynchronize();
    cudaMemcpy(&cpuProbeCycles, gpuProbeCycles, sizeof(long long), cudaMemcpyDeviceToHost);
    double floodAverage = (double) cpuProbeCycles/READS;

    //decode both by threshold and print
    cout << "Average 1: " << quietAverage << ", Average 2: " << floodAverage << endl;
    cout << "sent 0 -> decoded " << (quietAverage > THRESHOLD ? 1 : 0) << endl;
    cout << "sent 1 -> decoded " << (floodAverage > THRESHOLD ? 1 : 0) << endl;
}
