//BER-test transmitter: sends known pseudo-random bits (shared seed) so rx_ber can count errors. usage: ./tx_ber <slot_ns> <nbits>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <cstdlib>
const int FLOOD_RING_SIZE = 16 * 1024 * 1024;
const int FLOOD_STEPS = 10000;
const unsigned SEED = 1234;         //must match rx_ber for identical "random" bits

using namespace std;

//shared CLOCK_MONOTONIC clock in ns
unsigned long long cpuCurrentTime() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
//chases the ring to churn L2 (send a 1)
__global__ void floodKernel(int* ring, int steps, int* sink) {
    int idx = 0; for (int i = 0; i < steps; i++) idx = ring[idx]; sink[0] = idx;
}

int main(int argc, char** argv) {
    //parse slot length and bit count
    long long SLOT_NS = atoll(argv[1]);
    int NBITS = atoi(argv[2]);

    //build the flood ring and upload it
    int* cpuFloodRing = new int[FLOOD_RING_SIZE];
    for (int i = 0; i < FLOOD_RING_SIZE; i++) cpuFloodRing[i] = (i + 32) % FLOOD_RING_SIZE;
    int *gpuFloodRing, *gpuSink;
    cudaMalloc((void**)&gpuFloodRing, (size_t)FLOOD_RING_SIZE * sizeof(int));
    cudaMalloc((void**)&gpuSink, sizeof(int));
    cudaMemcpy(gpuFloodRing, cpuFloodRing, (size_t)FLOOD_RING_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    //generate the known bit string from the shared seed
    srand(SEED);
    int* messageBits = new int[NBITS];
    for (int i = 0; i < NBITS; i++) messageBits[i] = rand() & 1;

    //start 3 s out and print the start time for rx
    unsigned long long startTime = cpuCurrentTime() + 3000000000ULL;
    cout << startTime << endl;

    //for each slot: flood the whole slot for a 1, stay idle for a 0
    for (int k = 0; k < NBITS; k++) {
        unsigned long long slotStart = startTime + (unsigned long long)k       * SLOT_NS;
        unsigned long long slotEnd   = startTime + (unsigned long long)(k + 1) * SLOT_NS;
        while (cpuCurrentTime() < slotStart) {}
        if (messageBits[k] == 1) {
            while (cpuCurrentTime() < slotEnd) { floodKernel<<<1,1>>>(gpuFloodRing, FLOOD_STEPS, gpuSink); cudaDeviceSynchronize(); }
        } else {
            while (cpuCurrentTime() < slotEnd) {}
        }
    }
    cudaFree(gpuFloodRing); cudaFree(gpuSink); delete[] cpuFloodRing; delete[] messageBits;
    return 0;
}
