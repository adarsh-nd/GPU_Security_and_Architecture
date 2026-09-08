#include <iostream>
#include <cuda_runtime.h>
#include "../common/kernels.cuh"

const int PROBE_RING_SIZE = 1024 * 1024;
const int READS = 8192;
const int THRESHOLD = 400;
const long JAMMER_FLOATS = 64L * 1024 * 1024;
const int THREADS_PER_BLOCK = 256;
const int CAP_BLOCKS = 8;            //jammer intensity ceiling — kills the channel at ~6% honest cost
const int CONFIRM_WINDOW = 5;        //recent samples that must all agree before we respond
const int ITERS = 500;
const int RAMP_STEP = 2;
const int HOLD_SAMPLES = 40;

using namespace std;

//prime + probe the ring; returns avg cycles/read (high = an attacker is flooding L2)
double senseCache(int* gpuProbeRing, long long* gpuProbeCycles, int* gpuSink) {
    chaseKernel<<<1,1>>>(gpuProbeRing, READS, gpuSink);
    probeKernel<<<1,1>>>(gpuProbeRing, READS, gpuProbeCycles, gpuSink);
    cudaDeviceSynchronize();
    long long cpuProbeCycles;
    cudaMemcpy(&cpuProbeCycles, gpuProbeCycles, sizeof(long long), cudaMemcpyDeviceToHost);
    return (double)cpuProbeCycles/READS;
}

//true only if every recent sample was above threshold; the persistence gate that kills false alarms
bool confirmThreat(double* recentAverages, int windowSize) {
    for(int i = 0; i < windowSize; i++)
    {
        if(recentAverages[i] < THRESHOLD)
            return false;
    }
    return true;
}

//flood L2 at the given intensity to drown the attacker's channel
void runJammer(float* gpuInput, float* gpuOutput, long numFloats, int intensityBlocks, int threadsPerBlock) {
    streamKernel<<<intensityBlocks, threadsPerBlock>>>(gpuInput, gpuOutput, numFloats);
}

//nudge the jammer intensity up or down by one step, clamped to [0, CAP_BLOCKS]
int rampIntensity(int currentIntensity, bool threatConfirmed) {
    if(threatConfirmed)
        currentIntensity += RAMP_STEP;
    else
        currentIntensity -= RAMP_STEP;

    if(currentIntensity < 0)
        currentIntensity = 0;
    else if(currentIntensity > CAP_BLOCKS)
        currentIntensity = CAP_BLOCKS;

    return currentIntensity;
}

int main(int argc, char** argv) {
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

    //build the jammer's flood buffer and upload it
    float* cpuJammerInput = new float[JAMMER_FLOATS];
    for(int i = 0; i < JAMMER_FLOATS; i++)
    {
        cpuJammerInput[i] = (i + 32) % PROBE_RING_SIZE;
    }
    float* gpuJammerInput;
    float* gpuJammerOutput;
    cudaMalloc((void**)&gpuJammerInput, JAMMER_FLOATS * sizeof(float));
    cudaMemcpy(gpuJammerInput, cpuJammerInput, JAMMER_FLOATS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&gpuJammerOutput, CAP_BLOCKS * THREADS_PER_BLOCK * sizeof(float));

    //rolling window of recent senses, plus the persistent ramp and hold-timer state
    double* recentAverages = new double[CONFIRM_WINDOW];
    int intensity = 0;
    int clampTimer = 0;
    for(int i = 0; i < CONFIRM_WINDOW; i++)
    {
        recentAverages[i] = 0;
    }

    //run with any arg to disable jamming (sense-only control)
    bool jamEnabled = (argc < 2);
    int count = 0;

    //main loop: sense -> confirm -> latch a hold -> ramp intensity -> jam (or idle)
    while(true)
    {
        double average = senseCache(gpuProbeRing, gpuProbeCycles, gpuSink);
        recentAverages[count % CONFIRM_WINDOW] = average;
        bool confirmed = confirmThreat(recentAverages, CONFIRM_WINDOW);
        if(confirmed)
            clampTimer = HOLD_SAMPLES;
        else if(clampTimer > 0)
            clampTimer--;
        bool active = (clampTimer > 0);
        intensity = rampIntensity(intensity, active);
        if(intensity > 0 && jamEnabled)
        {
            runJammer(gpuJammerInput, gpuJammerOutput, JAMMER_FLOATS, intensity, THREADS_PER_BLOCK);
            cout << "Jamming, Intensity: " << intensity << endl;
        }
        else
        {
            cout << "Idle" << endl;
        }
        count++;
    }

    //cleanup (unreachable: the guard is a daemon, killed with Ctrl+C)
    cudaFree(gpuProbeRing);
    cudaFree(gpuProbeCycles);
    cudaFree(gpuSink);
    cudaFree(gpuJammerInput);
    cudaFree(gpuJammerOutput);
    delete[] cpuProbeRing;
    delete[] cpuJammerInput;
    delete[] recentAverages;
}
