#include <iostream>
#include <cuda_runtime.h>
#include <fstream>

using namespace std;

//Shares memory across 32 threads with spacing dictated by stride; access slowdowns with multiple threads/bank; used to reveal the # banks
__global__ void bankKernel(long long* cycles, int stride) {
    __shared__ float scratchpad[1024];
    int threadId = threadIdx.x;                             
    for (int i = threadId; i < 1024; i += 32) 
        scratchpad[i] = (float)i;
    __syncthreads();
    float sum = 0.0f;
    long long startCycle = clock64();
    for (int r = 0; r < 1000; r++) 
        sum += scratchpad[(threadId * stride + r) & 1023];  
    long long stopCycle = clock64();
    cycles[threadId] = (stopCycle - startCycle) + (sum < 0 ? 1 : 0);        
}

int main() {
    //creates and allocated the gpuCycles, 32 long longs
    long long* gpuCycles;
    cudaMalloc((void**)&gpuCycles, 32 * sizeof(long long));

    ofstream f("results/bank_conflicts.csv"); f << "stride,cycles\n";
    int strideArray[] = {1, 2, 4, 8, 16, 32};

    //runs the bank kernel for increasing stride sizes, copies gpu result to cpu and prnts the result
    long long cpuCycles[32];
    for (int i = 0; i < 6; i++)
    {
        int stride = strideArray[i];
        bankKernel<<<1, 32>>>(gpuCycles, stride);
        cudaDeviceSynchronize();
        cudaMemcpy(cpuCycles, gpuCycles, 32 * sizeof(long long), cudaMemcpyDeviceToHost);
        cout << "stride " << stride << ": " << cpuCycles[0] << " cycles" << endl;
        f << stride << "," << cpuCycles[0] << "\n";
    }

    f.close();
    cudaFree(gpuCycles);
    return 0;
}
