#include <iostream>
#include <cuda_runtime.h>
#include <unistd.h>
#include "../common/kernels.cuh"

using namespace std;

//spins on busy work and times it in both cycles (clock64) and ns (globaltimer); the ratio is the SM clock, used to check if a neighbor's load drags it down
__global__ void freqKernel(unsigned long long* timing) {
    long long startCycle = clock64();
    unsigned long long startNs = gpuCurrentTime();
    volatile long long busyWork = 0;
    for (long long i = 0; i < 20000000; i++)
    {
        busyWork += i;
    }
    long long endCycle = clock64();
    unsigned long long endNs = gpuCurrentTime();
    timing[0] = (unsigned long long)(endCycle - startCycle);
    timing[1] = endNs - startNs;
}

int main() {
    //allocate the timing buffer
    unsigned long long cpuTiming[2];
    unsigned long long* gpuTiming;
    cudaMalloc((void**)&gpuTiming, 2 * sizeof(unsigned long long));

    //sample the clock 32 times, printing cycles-per-ns, pausing between samples
    for(int i = 0; i < 32; i++)
    {
        freqKernel<<<1,1>>>(gpuTiming);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuTiming, gpuTiming, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cout << "Number of cycles per ns: " << (double)cpuTiming[0] / cpuTiming[1] << endl;
        usleep(100000);
    }
    cudaFree(gpuTiming);
}
