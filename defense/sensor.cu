#include <iostream>
#include <cuda_runtime.h>
#include "../common/kernels.cuh"

const int PROBE_RING_SIZE = 1024 * 1024;
const int READS = 8192;
const int THRESHOLD = 400;
const int ITERS = 500;

using namespace std;

//primes the ring into L2 then times a re-read; returns avg cycles/read (high = cache got flooded)
double probeCache(int* gpuProbeRing, long long* gpuProbeCycles, int* gpuSink) {
    chaseKernel<<<1,1>>>(gpuProbeRing, READS, gpuSink);
    probeKernel<<<1,1>>>(gpuProbeRing, READS, gpuProbeCycles, gpuSink);
    cudaDeviceSynchronize();
    long long cpuProbeCycles;
    cudaMemcpy(&cpuProbeCycles, gpuProbeCycles, sizeof(long long), cudaMemcpyDeviceToHost);
    return (double)cpuProbeCycles/READS;
}

//trips when the probe was slow, meaning the ring got evicted by a flood
bool isAlarm(double averageCycles) {
    return (averageCycles > THRESHOLD);
}

int main() {
    //build the probe ring and upload it to the GPU
    int* cpuProbeRing = new int[PROBE_RING_SIZE];
    for(int i = 0; i < PROBE_RING_SIZE; i++)
    {
        cpuProbeRing[i] = (i + 32) % PROBE_RING_SIZE;
    }
    int* gpuProbeRing;
    cudaMalloc((void**)&gpuProbeRing, PROBE_RING_SIZE * sizeof(int));
    cudaMemcpy(gpuProbeRing, cpuProbeRing, PROBE_RING_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    //scratch outputs for the probe kernels
    int* gpuSink;
    long long* gpuProbeCycles;
    cudaMalloc((void**)&gpuSink, sizeof(int));
    cudaMalloc((void**)&gpuProbeCycles, sizeof(long long));

    //probe 500 times, count how many trip the alarm, then print the alarm rate as a %
    int alarmCounter = 0;
    for(int i = 0; i <  ITERS; i++)
    {
        if(isAlarm(probeCache(gpuProbeRing, gpuProbeCycles, gpuSink)))
            alarmCounter++;
    }
    cout << alarmCounter * 100.0 / ITERS << endl;

    //free device + host memory
    cudaFree(gpuProbeRing);
    cudaFree(gpuProbeCycles);
    cudaFree(gpuSink);
    delete[] cpuProbeRing;
}
