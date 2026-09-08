#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

const int FLOOD_RING_SIZE = 16 * 1024 * 1024;   //flood ring = 64 MB (bigger than the 32 MB L2)
const int FLOOD_STEPS = 10000;                  //shorter bursts -> less overshoot into the next slot
const long long SLOT_NS = 150000000;            //150 ms per slot (must match rx.cu)
const int NUM_SLOTS = 48;

using namespace std;

//shared CLOCK_MONOTONIC clock in ns; tx and rx agree on time without touching the GPU
unsigned long long cpuCurrentTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

//chases the big ring, churning L2 to evict other programs' data; this is how we "send a 1"
__global__ void floodKernel(int* ring, int steps, int* sink) {
    int idx = 0;
    for (int i = 0; i < steps; i++) idx = ring[idx];
    sink[0] = idx;
}

int main() {
    //build the flood ring and upload it
    int* cpuFloodRing = new int[FLOOD_RING_SIZE];
    for (int i = 0; i < FLOOD_RING_SIZE; i++) cpuFloodRing[i] = (i + 32) % FLOOD_RING_SIZE;

    int* gpuFloodRing;
    int* gpuSink;
    cudaMalloc((void**)&gpuFloodRing, (size_t)FLOOD_RING_SIZE * sizeof(int));
    cudaMalloc((void**)&gpuSink, sizeof(int));
    cudaMemcpy(gpuFloodRing, cpuFloodRing, (size_t)FLOOD_RING_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    //encode "HOOKEM" into 48 bits, MSB-first
    const char* message = "HOOKEM";
    int messageBits[48];
    for (int c = 0; c < 6; c++)
    {
        for (int j = 0; j < 8; j++)
        {
            messageBits[c*8 + j] = (message[c] >> (7 - j)) & 1;
        }
    }

    //pick a start time 10 s out and print it so rx can lock to the same slot 0
    unsigned long long startTime = cpuCurrentTime() + 10000000000ULL;
    cout << "Starting Time: " << startTime << endl;

    //for each slot: wait for it, then flood the whole slot to send a 1 or stay idle for a 0
    for (int k = 0; k < NUM_SLOTS; k++) {
        unsigned long long slotStart = startTime + (unsigned long long)k       * SLOT_NS;
        unsigned long long slotEnd   = startTime + (unsigned long long)(k + 1) * SLOT_NS;

        while (cpuCurrentTime() < slotStart) {}

        if (messageBits[k] == 1) {
            //send a 1: flood the cache continuously for the whole slot
            while (cpuCurrentTime() < slotEnd) {
                floodKernel<<<1,1>>>(gpuFloodRing, FLOOD_STEPS, gpuSink);
                cudaDeviceSynchronize();
            }
        } else {
            //send a 0: leave the GPU idle for this slot
            while (cpuCurrentTime() < slotEnd) {}
        }
    }

    cudaFree(gpuFloodRing);
    cudaFree(gpuSink);
    delete[] cpuFloodRing;
    return 0;
}
