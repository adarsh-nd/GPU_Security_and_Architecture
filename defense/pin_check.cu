#include <iostream>
#include <cuda_runtime.h>
#include "../common/kernels.cuh"

const int RING_INTS = 1024 * 1024;                            //4 MB victim ring
const size_t PERSIST_BYTES = (size_t)RING_INTS * sizeof(int); //bytes to pin
const int PROBE_STEPS = 8192;
const int WARM_STEPS = 8192;
const int THREADS_PER_BLOCK = 256;
const long NUM_FLOATS = 64L * 1024 * 1024;                    //256 MB flood set: >> L2
const int FLOOD_BLOCKS = 8;
const int ROUNDS = 50;

using namespace std;

//pin the ring in L2: reserve a persisting slice and tag this stream's accesses to the ring as persisting
void pinRingInL2(int* gpuProbeRing, size_t bytes, cudaStream_t stream) {
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes);
    cudaAccessPolicyWindow window = {};
    window.base_ptr  = gpuProbeRing;
    window.num_bytes = bytes;
    window.hitRatio  = 1.0;
    window.hitProp   = cudaAccessPropertyPersisting;
    window.missProp  = cudaAccessPropertyStreaming;
    cudaStreamAttrValue value = {};
    value.accessPolicyWindow = window;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &value);
}

//time PROBE_STEPS reads of the ring on this stream; returns avg cycles/read
double probeCache(int* gpuProbeRing, long long* gpuProbeCycles, int* gpuSink, cudaStream_t stream) {
    probeKernel<<<1,1,0,stream>>>(gpuProbeRing, PROBE_STEPS, gpuProbeCycles, gpuSink);
    cudaStreamSynchronize(stream);

    long long cpuCycles;
    cudaMemcpy(&cpuCycles, gpuProbeCycles, sizeof(long long), cudaMemcpyDeviceToHost);
    return (double) cpuCycles/PROBE_STEPS;
}

int main() {
    //query the device's persisting-L2 limits; bail if unsupported
    int l2Size, maxPersist, maxWindow;
    cudaDeviceGetAttribute(&l2Size, cudaDevAttrL2CacheSize, 0);
    cudaDeviceGetAttribute(&maxPersist, cudaDevAttrMaxPersistingL2CacheSize,  0);
    cudaDeviceGetAttribute(&maxWindow, cudaDevAttrMaxAccessPolicyWindowSize, 0);
    cout << "L2=" << l2Size << " maxPersist=" << maxPersist << " maxWindow=" << maxWindow << endl;
    if(maxPersist == 0)
    {
        cout << "persisting L2 unsupported" << endl;
        return 0;
    }

    //build the victim ring and upload it
    int* cpuRing = new int[RING_INTS];
    for(int i = 0; i < RING_INTS; i++)
    {
        cpuRing[i] = (i + 32) % RING_INTS;
    }
    int* gpuRing;
    cudaMalloc((void**)&gpuRing, RING_INTS * sizeof(int));
    cudaMemcpy(gpuRing, cpuRing, RING_INTS * sizeof(int), cudaMemcpyHostToDevice);

    //scratch outputs for the probe
    int* gpuSink;
    cudaMalloc((void**)&gpuSink, sizeof(int));
    long long* gpuProbeCycles;
    cudaMalloc((void**)&gpuProbeCycles, sizeof(long long));

    //flood buffer (256 MB, >> L2) and its output
    float* gpuFlood;
    cudaMalloc((void**)&gpuFlood, NUM_FLOATS * sizeof(float));
    cudaMemset(gpuFlood, 0, NUM_FLOATS * sizeof(float));
    float* gpuFloodOut;
    cudaMalloc((void**)&gpuFloodOut, (long)FLOOD_BLOCKS * THREADS_PER_BLOCK * sizeof(float));

    //two non-blocking streams so victim and flood co-reside
    cudaStream_t sVictim, sAgg;
    cudaStreamCreateWithFlags(&sVictim, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&sAgg, cudaStreamNonBlocking);

    //condition A - unpinned: prime, flood, probe (expect the ring gets evicted / slow)
    chaseKernel<<<1,1,0,sVictim>>>(gpuRing, WARM_STEPS, gpuSink);
    cudaStreamSynchronize(sVictim);
    floodAggressor<<<FLOOD_BLOCKS, THREADS_PER_BLOCK, 0, sAgg>>>(gpuFlood, gpuFloodOut, NUM_FLOATS, ROUNDS);
    double cyclesNoPin = probeCache(gpuRing, gpuProbeCycles, gpuSink, sVictim);
    cudaStreamSynchronize(sAgg);

    //condition B - pinned: pin the ring, prime it into the persisting slice, flood, probe
    pinRingInL2(gpuRing, PERSIST_BYTES, sVictim);
    chaseKernel<<<1,1,0,sVictim>>>(gpuRing, WARM_STEPS, gpuSink);
    cudaStreamSynchronize(sVictim);
    floodAggressor<<<FLOOD_BLOCKS, THREADS_PER_BLOCK, 0, sAgg>>>(gpuFlood, gpuFloodOut, NUM_FLOATS, ROUNDS);
    double cyclesPinned = probeCache(gpuRing, gpuProbeCycles, gpuSink, sVictim);
    cudaStreamSynchronize(sAgg);

    //pinned << unpinned means the pin protected the ring; roughly equal means it did not
    cout << "unpinned=" << cyclesNoPin << " pinned=" << cyclesPinned << endl;

    //cleanup
    cudaFree(gpuRing);
    cudaFree(gpuProbeCycles);
    cudaFree(gpuSink);
    cudaFree(gpuFlood);
    cudaFree(gpuFloodOut);
    delete[] cpuRing;
}
