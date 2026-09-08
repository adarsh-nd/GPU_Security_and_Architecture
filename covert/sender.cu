#include <iostream>
#include <cuda_runtime.h>

const int FLOOD_RING_SIZE = 16 * 1024 * 1024;
const int FLOOD_STEPS = 1024 * 1024;

using namespace std;

//chases the big flood ring, churning L2 to evict other programs' data; this is how we "send a 1"
__global__ void primeKernel(int* ring, int steps, int* sink) {
    int idx = 0;
    for (int i = 0; i < steps; i++) {
        idx = ring[idx];
    }
    sink[0] = idx;
}

int main() {
    //build the flood ring and upload it
    int* cpuFloodRing = new int[FLOOD_RING_SIZE];
    for (int i = 0; i < FLOOD_RING_SIZE; i++) {
        cpuFloodRing[i] = (i + 32) % FLOOD_RING_SIZE;
    }

    int* gpuFloodRing;
    int* gpuSink;
    cudaMalloc((void**)&gpuFloodRing, (size_t)FLOOD_RING_SIZE * sizeof(int));
    cudaMalloc((void**)&gpuSink, sizeof(int));
    cudaMemcpy(gpuFloodRing, cpuFloodRing, (size_t)FLOOD_RING_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    //flood L2 forever; each launch churns the cache until you kill it
    while (true) {
        primeKernel<<<1, 1>>>(gpuFloodRing, FLOOD_STEPS, gpuSink);
        cudaDeviceSynchronize();
    }

    //cleanup (unreachable: the sender is a daemon, killed with Ctrl+C)
    cudaFree(gpuFloodRing);
    cudaFree(gpuSink);
    delete[] cpuFloodRing;
    return 0;
}
