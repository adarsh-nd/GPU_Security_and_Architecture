#include <iostream>
#include <cuda_runtime.h>
#include "../common/kernels.cuh"

const int PROBE_RING_SIZE = 1024 * 1024;   //4 MB canary in the shared L2
const int READS = 8192;
const int THRESHOLD = 400;                  //> 400 cyc means the ring was evicted, i.e. victim was busy
const long PHASE_NS = 500L * 1000000;       //must match victim.cu
const int CYCLES = 20;                       //must match victim.cu

using namespace std;

//time one probe of the canary; returns avg cycles/read (high = the victim evicted it by doing work)
double probeCache(int* gpuProbeRing, long long* gpuProbeCycles, int* gpuSink) {
    probeKernel<<<1,1>>>(gpuProbeRing, READS, gpuProbeCycles, gpuSink);
    cudaDeviceSynchronize();
    long long cpuProbeCycles;
    cudaMemcpy(&cpuProbeCycles, gpuProbeCycles, sizeof(long long), cudaMemcpyDeviceToHost);
    return (double)cpuProbeCycles / READS;
}

int main(int argc, char** argv) {
    //parse the victim's start time, used to align ground truth
    if(argc < 2)
    {
        cout << "usage: ./spy <victim_start_time>" << endl;
        return 1;
    }
    unsigned long long startTime = atoll(argv[1]);

    //build the probe ring and upload it
    int* cpuProbeRing = new int[PROBE_RING_SIZE];
    for(int i = 0; i < PROBE_RING_SIZE; i++)
    {
        cpuProbeRing[i] = (i + 32) % PROBE_RING_SIZE;
    }
    int* gpuProbeRing;
    cudaMalloc((void**)&gpuProbeRing, PROBE_RING_SIZE * sizeof(int));
    cudaMemcpy(gpuProbeRing, cpuProbeRing, PROBE_RING_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    //scratch outputs for the probe
    int* gpuSink;
    long long* gpuProbeCycles;
    cudaMalloc((void**)&gpuSink, sizeof(int));
    cudaMalloc((void**)&gpuProbeCycles, sizeof(long long));

    //prime the canary once
    chaseKernel<<<1,1>>>(gpuProbeRing, READS, gpuSink);
    cudaDeviceSynchronize();

    unsigned long long endTime = startTime + (unsigned long long)(2 * CYCLES) * PHASE_NS;

    int correct = 0;
    int total = 0;
    double busySum = 0;
    double idleSum = 0;
    int busyN = 0;
    int idleN = 0;

    //spy until the victim's schedule ends: probe, guess busy/idle, compare to the known schedule, tally
    while(cpuCurrentTime() < endTime)
    {
        double average = probeCache(gpuProbeRing, gpuProbeCycles, gpuSink);
        unsigned long long now = cpuCurrentTime();

        unsigned long long halfPhase = (now - startTime) / PHASE_NS;   //which 500 ms half we are in
        bool truthBusy = (halfPhase % 2 == 0);                          //even halves = victim busy
        bool inferBusy = (average > THRESHOLD);

        if(truthBusy == inferBusy)
            correct++;
        total++;

        if(truthBusy)
        {
            busySum += average;
            busyN++;
        }
        else
        {
            idleSum += average;
            idleN++;
        }
    }

    //print sample count, busy/idle latency averages, and detection accuracy
    cout << "samples=" << total
         << " avg_busy=" << busySum / busyN
         << " avg_idle=" << idleSum / idleN
         << " accuracy=" << (correct * 100.0 / total) << "%" << endl;

    //cleanup
    cudaFree(gpuProbeRing);
    cudaFree(gpuProbeCycles);
    cudaFree(gpuSink);
    delete[] cpuProbeRing;
}
