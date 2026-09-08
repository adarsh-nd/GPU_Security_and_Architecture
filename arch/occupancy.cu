#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include "../common/kernels.cuh"

using namespace std;

int main() {
    //allocate and fill the input, copy it to the GPU
    long numFloats = 64L * 1024 * 1024;
    float* cpuInput = new float[numFloats];
    float* gpuInput;
    float* gpuOutput;

    for (long i = 0; i < numFloats; i++)
    {
        cpuInput[i] = 1.0f;
    }
    cudaMalloc((void**)&gpuInput, (size_t)numFloats * sizeof(float));
    cudaMalloc((void**)&gpuOutput, (size_t)1024 * 32 * sizeof(float));
    cudaMemcpy(gpuInput, cpuInput, (size_t)numFloats * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;                       //32 threads = 1 warp per block, so warps == blocks
    ofstream f("measurements/occupancy.csv");
    f << "warps,gbps\n";

    //create the timing events once, reused each iteration
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    //sweep the block count (= warps): warm-up run, then a timed run, record GB/s
    int blockCounts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (int b = 0; b < 11; b++) {
        int blocks = blockCounts[b];
        streamKernel<<<blocks, threadsPerBlock>>>(gpuInput, gpuOutput, numFloats);
        cudaDeviceSynchronize();
        cudaEventRecord(startEvent);
        streamKernel<<<blocks, threadsPerBlock>>>(gpuInput, gpuOutput, numFloats);
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        float elapsedMs = 0; cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);

        double bandwidthGBps = (double)numFloats * sizeof(float) / (elapsedMs / 1000.0) / 1e9;
        cout << blocks << " Warps, " << bandwidthGBps << " GB/s" << endl;
        f << blocks << "," << bandwidthGBps << "\n";
    }

    //cleanup
    f.close();
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(gpuInput);
    cudaFree(gpuOutput);
    delete[] cpuInput;
    return 0;
}
