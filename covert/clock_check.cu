#include <iostream>
#include <cuda_runtime.h>
#include <fstream>

const int NUM_SAMPLES = 1000;
using namespace std;

//reads the GPU global timer 1000 times back-to-back; used to check its resolution and that two processes see the same clock
__global__ void readClockKernel(unsigned long long* clockSamples) {
    for(int i = 0; i < NUM_SAMPLES; i++) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        clockSamples[i] = t;
    }
}

int main() {
    //allocate the sample buffer
    unsigned long long* cpuClockSamples = new unsigned long long[NUM_SAMPLES];
    unsigned long long* gpuClockSamples;
    cudaMalloc((void**)&gpuClockSamples, NUM_SAMPLES * sizeof(unsigned long long));

    //read the clock 1000 times on the GPU, copy back, print every sample
    readClockKernel<<<1,1>>>(gpuClockSamples);
    cudaDeviceSynchronize();
    cudaMemcpy(cpuClockSamples, gpuClockSamples, NUM_SAMPLES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    for(int i = 0; i < NUM_SAMPLES; i++)
    {
        cout << "Time "<< i << ": " << cpuClockSamples[i] << endl;
    }

    //cleanup
    cudaFree(gpuClockSamples);
    delete[] cpuClockSamples;
}
