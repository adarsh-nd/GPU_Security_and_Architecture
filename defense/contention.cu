#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "../common/kernels.cuh"

const int RING_INTS = 1024 * 1024;        //4 MB victim ring: > L1 (evictable), << L2
const int PROBE_STEPS = 8192;
const int WARM_STEPS = 8192;
const int THREADS_PER_BLOCK = 256;
const long NUM_FLOATS = 64L * 1024 * 1024; //256 MB flood set: >> L2
const int SMALL_COUNT = 1024;              //4 KB control set: fits L1, ~0 L2 pressure

using namespace std;

//co-tenant control: same launch and load count as the flood but hammers a tiny L1-resident buffer, so it burns GPU without touching L2 (isolates occupancy from cache)
__global__ void matchedControl(const float* smallBuf, float* output, long numLoads, int rounds, int smallCount) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)gridDim.x * blockDim.x;
    float sum = 0.0f;

    for(int r = 0; r < rounds; r++)
    {
        for(long i = tid; i < numLoads; i += total)
        {
            sum += smallBuf[(i + tid) % smallCount];
        }
    }

    output[tid] = sum;
}

int main(int argc, char** argv) {
    //parse args: mode, aggressor block count, rounds
    if(argc < 3)
    {
        cout << "usage: ./contention <none|flood|control> <aggBlocks> [rounds]" << endl;
        return 1;
    }
    string mode = argv[1];
    int aggBlocks = atoi(argv[2]);
    int rounds = (argc > 3) ? atoi(argv[3]) : 50;
    if(aggBlocks < 1) aggBlocks = 1;

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
    long long* gpuCycles;
    cudaMalloc((void**)&gpuSink, sizeof(int));
    cudaMalloc((void**)&gpuCycles, sizeof(long long));

    //flood buffer (256 MB, >> L2)
    float* cpuFlood = new float[NUM_FLOATS];
    for(long i = 0; i < NUM_FLOATS; i++)
    {
        cpuFlood[i] = 1.0f;
    }
    float* gpuFlood;
    cudaMalloc((void**)&gpuFlood, NUM_FLOATS * sizeof(float));
    cudaMemcpy(gpuFlood, cpuFlood, NUM_FLOATS * sizeof(float), cudaMemcpyHostToDevice);

    //control buffer (4 KB, L1-resident)
    float* cpuSmall = new float[SMALL_COUNT];
    for(int i = 0; i < SMALL_COUNT; i++)
    {
        cpuSmall[i] = 1.0f;
    }
    float* gpuSmall;
    cudaMalloc((void**)&gpuSmall, SMALL_COUNT * sizeof(float));
    cudaMemcpy(gpuSmall, cpuSmall, SMALL_COUNT * sizeof(float), cudaMemcpyHostToDevice);

    //aggressor output
    float* gpuAggOutput;
    cudaMalloc((void**)&gpuAggOutput, (long)aggBlocks * THREADS_PER_BLOCK * sizeof(float));

    //two non-blocking streams so victim and aggressor can co-reside
    cudaStream_t sVictim, sAgg;
    cudaStreamCreateWithFlags(&sVictim, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&sAgg, cudaStreamNonBlocking);

    //prime the victim ring into L2
    chaseKernel<<<1, 1, 0, sVictim>>>(gpuRing, WARM_STEPS, gpuSink);
    cudaStreamSynchronize(sVictim);

    //launch the chosen aggressor on its own stream, then probe the victim while it runs
    if(mode == "flood")
        floodAggressor<<<aggBlocks, THREADS_PER_BLOCK, 0, sAgg>>>(gpuFlood, gpuAggOutput, NUM_FLOATS, rounds);
    else if(mode == "control")
        matchedControl<<<aggBlocks, THREADS_PER_BLOCK, 0, sAgg>>>(gpuSmall, gpuAggOutput, NUM_FLOATS, rounds, SMALL_COUNT);

    probeKernel<<<1, 1, 0, sVictim>>>(gpuRing, PROBE_STEPS, gpuCycles, gpuSink);

    cudaStreamSynchronize(sVictim);
    cudaStreamSynchronize(sAgg);

    //read the probe timing and print avg cycles/read
    long long cpuCycles;
    cudaMemcpy(&cpuCycles, gpuCycles, sizeof(long long), cudaMemcpyDeviceToHost);
    cout << "MODE=" << mode << " BLOCKS=" << aggBlocks << " ROUNDS=" << rounds
         << " AVG_CYCLES=" << (double)cpuCycles / PROBE_STEPS << endl;

    //cleanup
    cudaFree(gpuRing);
    cudaFree(gpuSink);
    cudaFree(gpuCycles);
    cudaFree(gpuFlood);
    cudaFree(gpuSmall);
    cudaFree(gpuAggOutput);
    delete[] cpuRing;
    delete[] cpuFlood;
    delete[] cpuSmall;
}
