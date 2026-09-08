#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "../common/kernels.cuh"

const long JAMMER_FLOATS = 64L * 1024 * 1024;
const int THREADS_PER_BLOCK = 256;
const int CAP_BLOCKS = 768;
const long COMPUTE_ITERS = 5000000;

using namespace std;

//pure-compute co-tenant: grinds FMAs in registers touching ~0 memory, to load the GPU without evicting L2 (the occupancy-only control)
__global__ void computeKernel(int* gpuSink, long iters) {
    float x = threadIdx.x * 0.001f + blockIdx.x;
    for(long i = 0; i < iters; i++)
    {
        x = x * 1.0000001f + 0.5f;
    }
    if(x == -12345.0f)
        gpuSink[0] = (int)x;
}

int main(int argc, char** argv) {
    //parse mode
    if(argc < 2)
    {
        cout << "usage: ./pure_jammer <compute|stream>" << endl;
        return 1;
    }
    string mode = argv[1];

    if(mode == "stream")
    {
        //stream mode: flood a >> L2 buffer forever (the real jammer); optional arg sets intensity
        float* cpuJammerInput = new float[JAMMER_FLOATS];
        for(long i = 0; i < JAMMER_FLOATS; i++)
        {
            cpuJammerInput[i] = (float)(i % 1024);
        }
        float* gpuJammerInput;
        float* gpuJammerOutput;
        cudaMalloc((void**)&gpuJammerInput, JAMMER_FLOATS * sizeof(float));
        cudaMemcpy(gpuJammerInput, cpuJammerInput, JAMMER_FLOATS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&gpuJammerOutput, CAP_BLOCKS * THREADS_PER_BLOCK * sizeof(float));

        int jamBlocks = (argc > 2) ? atoi(argv[2]) : CAP_BLOCKS;
        while(true)
        {
            streamKernel<<<jamBlocks, THREADS_PER_BLOCK>>>(gpuJammerInput, gpuJammerOutput, JAMMER_FLOATS);
            cudaDeviceSynchronize();
        }
    }
    else if(mode == "compute")
    {
        //compute mode: occupy the SMs with register work, no L2 pressure
        int* gpuSink;
        cudaMalloc((void**)&gpuSink, sizeof(int));

        while(true)
        {
            computeKernel<<<CAP_BLOCKS, THREADS_PER_BLOCK>>>(gpuSink, COMPUTE_ITERS);
            cudaDeviceSynchronize();
        }
    }
    else
    {
        cout << "unknown mode: " << mode << endl;
        return 1;
    }
}
