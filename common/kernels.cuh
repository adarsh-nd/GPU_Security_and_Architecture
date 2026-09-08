#pragma once
#include <cuda_runtime.h>
#include <time.h>

//returns the current GPU global timer value
__device__ inline unsigned long long gpuCurrentTime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

//returns the current system-wide clock value
static inline unsigned long long cpuCurrentTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

//walks through an array serially to pull each line into L2, chaining indices to the previous value to prevent overlap
__global__ void chaseKernel(const int* ring, int steps, int* sink) {
    int idx = 0;

    for (int i = 0; i < steps; i++)
    {
        idx = ring[idx];
    }

    sink[0] = idx;
}

//chase kernel that times how long each read takes
__global__ void probeKernel(const int* ring, int reads, long long* cycles, int* sink) {
    int idx = 0;
    long long totalCycles = 0;

    for (int i = 0; i < reads; i++)
    {
        long long startCycle = clock64();
        idx = ring[idx];
        totalCycles += clock64() - startCycle;
    }

    cycles[0] = totalCycles;
    sink[0] = idx;
}

//all threads are coalesced to collectively read all data; used to benchmark bandwidth, to flood L2 for the jammer, and to measure honest workload
__global__ void streamKernel(const float* input, float* output, long numFloats) {
    long threadId = blockIdx.x * blockDim.x + threadIdx.x;
    long totalThreads = (long)gridDim.x * blockDim.x;
    float sum = 0.0f;

    for (long i = threadId; i < numFloats; i += totalThreads)
    {
        sum += input[i];
    }

    output[threadId] = sum;
}

//same as streamKernel but repeats it for # rounds, allows for overlap with victim's probe
__global__ void floodAggressor(const float* input, float* output, long numFloats, int rounds) {
    long threadId = blockIdx.x * blockDim.x + threadIdx.x;
    long totalThreads = (long)gridDim.x * blockDim.x;
    float sum = 0.0f;

    for(int r = 0; r < rounds; r++)
    {
        for(long i = threadId; i < numFloats; i += totalThreads)
        {
            sum += input[i];
        }
    }

    output[threadId] = sum;
}