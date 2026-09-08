#include <iostream>
#include <cuda_runtime.h>
#include "../common/kernels.cuh"

using namespace std;

int main() {
    long numFloats = 64L * 1024 * 1024;
    int threadsPerBlock = 256;
    int blocks = 768;
    float* cpuInput = new float[numFloats];
    float* gpuInput;
    float* gpuOutput;

    //fills cpuInput array with data
    for(long i = 0; i < numFloats; i++)
    {
        cpuInput[i] = 1.0f;
    }

    //gpu input and output allocation, copies cpu input into gpu input
    cudaMalloc((void**)&gpuInput, (size_t)numFloats * sizeof(float));
    cudaMalloc((void**)&gpuOutput, (size_t)blocks * threadsPerBlock * sizeof(float));
    cudaMemcpy(gpuInput, cpuInput, (size_t)numFloats * sizeof(float), cudaMemcpyHostToDevice);

    //steamKernel junk run, serves as a gpu warmup
    streamKernel<<<blocks, threadsPerBlock>>>(gpuInput, gpuOutput, numFloats);
    cudaDeviceSynchronize();

    //creates cuda events for the start and stop to time the streamKernel takes, waits for that timing to conclude before moving on
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent); 
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);
    streamKernel<<<blocks, threadsPerBlock>>>(gpuInput, gpuOutput, numFloats);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float elapsedMs = 0;
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);

    //Uses the time to measure the bandwidth
    double bandwidthGBps = (double)numFloats * sizeof(float) / (elapsedMs / 1000.0) / 1e9;
    cout << "Bandwidth Rate: " << bandwidthGBps << endl;
    cudaFree(gpuInput);
    cudaFree(gpuOutput);
    delete[] cpuInput;
}
