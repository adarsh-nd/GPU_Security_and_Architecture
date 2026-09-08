#include <iostream>
#include <cuda_runtime.h>
#include "../common/kernels.cuh"

const long NUM_FLOATS = 64L * 1024 * 1024;      //honest workload set
const long JAMMER_FLOATS = 64L * 1024 * 1024;   //jammer set
const int THREADS_PER_BLOCK = 256;
const int HONEST_BLOCKS = 768;                  //honest workload at full occupancy
const int ROUNDS = 100;                         //jammer duration: >> honest kernel, blankets it

using namespace std;

int main(int argc, char** argv) {
    //parse arg: jammer block count (0 = no jammer, the at-rest baseline)
    if(argc < 2)
    {
        cout << "usage: ./cost <jammerBlocks>   (0 = no jammer / at-rest baseline)" << endl;
        return 1;
    }
    int jammerBlocks = atoi(argv[1]);

    //honest workload buffers
    float* gpuHonestIn;
    float* gpuHonestOut;
    cudaMalloc((void**)&gpuHonestIn, NUM_FLOATS * sizeof(float));
    cudaMemset(gpuHonestIn, 0, NUM_FLOATS * sizeof(float));
    cudaMalloc((void**)&gpuHonestOut, (long)HONEST_BLOCKS * THREADS_PER_BLOCK * sizeof(float));

    //jammer buffers
    float* gpuJammerIn;
    float* gpuJammerOut;
    cudaMalloc((void**)&gpuJammerIn, JAMMER_FLOATS * sizeof(float));
    cudaMemset(gpuJammerIn, 0, JAMMER_FLOATS * sizeof(float));
    cudaMalloc((void**)&gpuJammerOut, (long)(jammerBlocks > 0 ? jammerBlocks : 1) * THREADS_PER_BLOCK * sizeof(float));

    //two non-blocking streams so the honest workload and jammer run concurrently
    cudaStream_t sHonest, sJammer;
    cudaStreamCreateWithFlags(&sHonest, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&sJammer, cudaStreamNonBlocking);

    //warm-up run of the honest workload
    streamKernel<<<HONEST_BLOCKS, THREADS_PER_BLOCK, 0, sHonest>>>(gpuHonestIn, gpuHonestOut, NUM_FLOATS);
    cudaStreamSynchronize(sHonest);

    //start the jammer (if any) so it blankets the timed run
    if(jammerBlocks > 0)
        floodAggressor<<<jammerBlocks, THREADS_PER_BLOCK, 0, sJammer>>>(gpuJammerIn, gpuJammerOut, JAMMER_FLOATS, ROUNDS);

    //time the honest run under the jammer and print its throughput
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, sHonest);
    streamKernel<<<HONEST_BLOCKS, THREADS_PER_BLOCK, 0, sHonest>>>(gpuHonestIn, gpuHonestOut, NUM_FLOATS);
    cudaEventRecord(stopEvent, sHonest);
    cudaEventSynchronize(stopEvent);

    float elapsedMs = 0;
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    double gbps = NUM_FLOATS * sizeof(float) / (elapsedMs / 1000.0) / 1e9;
    cout << "JAMMER_BLOCKS=" << jammerBlocks << " HONEST_MS=" << elapsedMs << " HONEST_GBPS=" << gbps << endl;

    //cleanup
    cudaStreamSynchronize(sJammer);
    cudaFree(gpuHonestIn);
    cudaFree(gpuHonestOut);
    cudaFree(gpuJammerIn);
    cudaFree(gpuJammerOut);
}
