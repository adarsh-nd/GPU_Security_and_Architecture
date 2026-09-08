#include <iostream>
#include <cuda_runtime.h>
#include "../common/kernels.cuh"

const long NUM_FLOATS = 64L * 1024 * 1024;
const int BLOCKS = 768;
const int THREADS_PER_BLOCK = 256;

using namespace std;

//times one streamKernel pass with cuda events; returns ms, used to measure honest throughput
float runWorkloadMs(float* gpuInput, float* gpuOutput, long numFloats, int blocks, int threadsPerBlock) {
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent); cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);
    streamKernel<<<blocks, threadsPerBlock>>>(gpuInput, gpuOutput, numFloats);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float elapsedMs = 0;
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    return elapsedMs;
}

int main() {
    //allocate and fill the input, copy it to the GPU
    float* cpuInput = new float[NUM_FLOATS];
    for(int i = 0; i < NUM_FLOATS; i++)
    {
        cpuInput[i] = 0.0f;
    }

    float* gpuInput;
    cudaMalloc((void**) &gpuInput, NUM_FLOATS * sizeof(float));
    cudaMemcpy(gpuInput, cpuInput, NUM_FLOATS * sizeof(float), cudaMemcpyHostToDevice);

    float* gpuOutput;
    cudaMalloc((void**)&gpuOutput, BLOCKS * THREADS_PER_BLOCK * sizeof(float));

    //warm-up run, then a timed run, print the workload time and bandwidth
    runWorkloadMs(gpuInput, gpuOutput, NUM_FLOATS, BLOCKS, THREADS_PER_BLOCK);
    float workloadMs = runWorkloadMs(gpuInput, gpuOutput, NUM_FLOATS, BLOCKS, THREADS_PER_BLOCK);
    cout << "Workload MS: " << workloadMs << ", Bandwidth (Gbps): " << NUM_FLOATS * sizeof(float) / (workloadMs/1000.0) / 1e9 << endl;

    //cleanup
    cudaFree(gpuInput);
    cudaFree(gpuOutput);
    delete[] cpuInput;
}
