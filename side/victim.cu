#include <iostream>
#include <cuda_runtime.h>
#include "../common/kernels.cuh"

const long NUM_FLOATS = 64L * 1024 * 1024;   //256 MB working set (>> L2, so a busy phase evicts the spy)
const int BLOCKS = 256;                       //a "real" GPU workload during busy phases
const int THREADS_PER_BLOCK = 256;
const long PHASE_NS = 500L * 1000000;         //500 ms busy, then 500 ms idle
const int CYCLES = 20;                         //20 busy/idle cycles = ~20 s total (must match spy.cu)

using namespace std;

int main() {
    //allocate the workload buffers
    float* gpuInput;
    float* gpuOutput;
    cudaMalloc((void**)&gpuInput, NUM_FLOATS * sizeof(float));
    cudaMemset(gpuInput, 0, NUM_FLOATS * sizeof(float));
    cudaMalloc((void**)&gpuOutput, (long)BLOCKS * THREADS_PER_BLOCK * sizeof(float));

    //publish the start time so the spy can align to the same schedule
    unsigned long long startTime = cpuCurrentTime();
    cout << startTime << endl;

    //alternate 500 ms busy (real GPU work) and 500 ms idle (CPU spin, GPU untouched), for CYCLES rounds
    for(int c = 0; c < CYCLES; c++)
    {
        unsigned long long busyStart = startTime + (unsigned long long)(2 * c) * PHASE_NS;
        unsigned long long busyEnd   = busyStart + PHASE_NS;
        unsigned long long idleEnd   = busyEnd + PHASE_NS;

        while(cpuCurrentTime() < busyStart) {}
        while(cpuCurrentTime() < busyEnd)
        {
            streamKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(gpuInput, gpuOutput, NUM_FLOATS);
            cudaDeviceSynchronize();
        }
        while(cpuCurrentTime() < idleEnd) {}
    }

    //cleanup
    cudaFree(gpuInput);
    cudaFree(gpuOutput);
}
