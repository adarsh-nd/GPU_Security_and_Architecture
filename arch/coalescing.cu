#include <iostream>
#include <cuda_runtime.h>
#include <fstream>

using namespace std;

//reads memory at a chosen stride; wide strides scatter each warp's reads across separate cache lines so bandwidth craters, revealing the coalescing penalty
__global__ void coalesceKernel(const float* input, float* output, long numFloats, int stride) {
    long threadId    = blockIdx.x * blockDim.x + threadIdx.x;
    long totalThreads = (long)gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (long i = threadId; i < numFloats; i += totalThreads)
        sum += input[(i * (long)stride) % numFloats];   //stride spacing, wrapped to stay in bounds
    output[threadId] = sum;
}

int main() {
    //allocate and fill the input, copy it to the GPU
    long numFloats = 64L * 1024 * 1024;
    float* cpuInput = new float[numFloats];
    float* gpuInput;
    float* gpuOutput;

    for (long i = 0; i < numFloats; i++)
    {
        cpuInput[i] = 1.0f;
    }

    int blocks = 768;
    int threadsPerBlock = 256;
    cudaMalloc((void**)&gpuInput, (size_t)numFloats * sizeof(float));
    cudaMalloc((void**)&gpuOutput, (size_t)blocks * threadsPerBlock * sizeof(float));
    cudaMemcpy(gpuInput, cpuInput, (size_t)numFloats * sizeof(float), cudaMemcpyHostToDevice);

    //open the csv and make the reusable timing events
    ofstream f("measurements/coalescing.csv");
    f << "stride,gbps\n";

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    //sweep the stride: warm-up run, then a timed run, record GB/s for each
    int strideArray[] = {1, 2, 4, 8, 16, 32, 64};
    for (int i = 0; i < 7; i++) {
        int stride = strideArray[i];

        coalesceKernel<<<blocks, threadsPerBlock>>>(gpuInput, gpuOutput, numFloats, stride);
        cudaDeviceSynchronize();

        cudaEventRecord(startEvent);
        coalesceKernel<<<blocks, threadsPerBlock>>>(gpuInput, gpuOutput, numFloats, stride);
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        float elapsedMs = 0; cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);

        double bandwidthGBps = (double)numFloats * sizeof(float) / (elapsedMs / 1000.0) / 1e9;
        cout << "stride " << stride << ": " << bandwidthGBps << " GB/s" << endl;
        f << stride << "," << bandwidthGBps << "\n";
    }

    //cleanup
    f.close();
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(gpuInput);
    cudaFree(gpuOutput);
    delete[] cpuInput;
    return 0;
}
