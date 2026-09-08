#include <iostream>
#include <string>
#include <unistd.h>
#include <cuda_runtime.h>
#include "../common/kernels.cuh"

const int RING_INTS = 1024 * 1024;
const size_t PERSIST_BYTES = (size_t)RING_INTS * sizeof(int);
const int PROBE_STEPS = 8192;
const int WARM_STEPS = 8192;
const int ITERS = 200;

using namespace std;

//pin the ring in L2: reserve a persisting slice and tag this stream's accesses to the ring as persisting
void pinRingInL2(int* gpuRing, size_t bytes, cudaStream_t stream) {
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes);
    cudaAccessPolicyWindow window = {};
    window.base_ptr  = gpuRing;
    window.num_bytes = bytes;
    window.hitRatio  = 1.0;
    window.hitProp   = cudaAccessPropertyPersisting;
    window.missProp  = cudaAccessPropertyStreaming;
    cudaStreamAttrValue value = {};
    value.accessPolicyWindow = window;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &value);
}

int main(int argc, char** argv) {
    //pinned mode when run with "pinned"
    bool pinned = (argc > 1 && string(argv[1]) == "pinned");

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
    long long* gpuCycles;
    cudaMalloc((void**)&gpuCycles, sizeof(long long));

    //one non-blocking stream, pinned if requested
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if(pinned) pinRingInL2(gpuRing, PERSIST_BYTES, stream);

    //prime the ring once
    chaseKernel<<<1,1,0,stream>>>(gpuRing, WARM_STEPS, gpuSink);
    cudaStreamSynchronize(stream);

    //probe in a loop with a 1 ms gap so a co-tenant process can evict between probes, average the latency
    double total = 0;
    for(int i = 0; i < ITERS; i++)
    {
        probeKernel<<<1,1,0,stream>>>(gpuRing, PROBE_STEPS, gpuCycles, gpuSink);
        cudaStreamSynchronize(stream);
        long long c;
        cudaMemcpy(&c, gpuCycles, sizeof(long long), cudaMemcpyDeviceToHost);
        total += (double)c / PROBE_STEPS;
        usleep(1000);
    }
    cout << (pinned ? "pinned" : "unpinned") << " avg_cycles=" << total / ITERS << endl;

    //cleanup
    cudaFree(gpuRing);
    cudaFree(gpuSink);
    cudaFree(gpuCycles);
    delete[] cpuRing;
}
