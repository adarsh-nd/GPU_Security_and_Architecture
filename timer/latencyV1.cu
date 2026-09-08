#include <iostream>
#include <cuda_runtime.h>
#include <fstream>

const int READS = 8192;

using namespace std;


__global__ void timingKernel(int* ring, long long* latencies, int* sink, int steps) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    for(int i = 0; i < READS; i++)
    {
        long long startCycle = clock64();
        idx = ring[idx];
        latencies[i] = clock64() - startCycle;
    }
    sink[0] = idx;
}

int main() {
    const int RING_SIZE = 1024 * 65536;
    size_t ringBytes = (size_t)(RING_SIZE * sizeof(int));
    int* cpuRing = new int[RING_SIZE]; //heap allocated
    int* gpuRing;
    cudaMalloc((void**)&gpuRing, ringBytes);

    int stride = 32;
    for(int i = 0; i < RING_SIZE; i++)
    {
        cpuRing[i] = (i + stride) % RING_SIZE;
    }

    cudaMemcpy(gpuRing, cpuRing, ringBytes, cudaMemcpyHostToDevice);

    long long* gpuLatencies;
    int* gpuSink;
    cudaMalloc((void**)&gpuLatencies, READS * sizeof(long long));
    cudaMalloc((void**)&gpuSink, READS * sizeof(int));
    timingKernel<<<1,1>>>(gpuRing, gpuLatencies, gpuSink, RING_SIZE);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        cout << "Error:" << cudaGetErrorString(error) << endl;
    }

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        cout << "Run:" << cudaGetErrorString(error) << endl;
    }

    long long* cpuLatencies = new long long[READS];
    cudaMemcpy(cpuLatencies, gpuLatencies, READS * sizeof(long long), cudaMemcpyDeviceToHost);
    ofstream f("results/latency_hits.csv");
    for(int i = 0; i < READS; i++)
    {
        f << cpuLatencies[i] << "\n";
    }
    f.close();

    delete[] cpuRing;
    delete[] cpuLatencies;
    cudaFree(gpuRing);
    cudaFree(gpuLatencies);
    cudaFree(gpuSink);
}
