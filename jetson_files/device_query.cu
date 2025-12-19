/*
 * Programa simple para verificar información de CUDA
 * Compilar: nvcc -o device_query device_query.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    printf("========================================\n");
    printf("   CUDA Device Query\n");
    printf("========================================\n\n");
    
    // Obtener número de dispositivos
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("Error: cudaGetDeviceCount failed!\n");
        printf("Error code: %d - %s\n", error, cudaGetErrorString(error));
        return 1;
    }
    
    printf("Number of CUDA devices: %d\n\n", deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA capable devices found!\n");
        return 1;
    }
    
    // Iterar sobre cada dispositivo
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", 
               prop.totalGlobalMem / 1073741824.0);
        printf("  Memory Clock Rate: %.2f GHz\n", 
               prop.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  L2 Cache Size: %.2f MB\n", prop.l2CacheSize / 1048576.0);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        printf("  Max Grid Size: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max Block Dimensions: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Warp Size: %d threads\n", prop.warpSize);
        printf("  Shared Memory per Block: %.2f KB\n", 
               prop.sharedMemPerBlock / 1024.0);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
        printf("  Concurrent Kernels: %s\n", 
               prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("  Unified Addressing: %s\n", 
               prop.unifiedAddressing ? "Yes" : "No");
        printf("  Compute Mode: ");
        
        switch (prop.computeMode) {
            case cudaComputeModeDefault:
                printf("Default (multiple threads can use)\n");
                break;
            case cudaComputeModeExclusive:
                printf("Exclusive (only one thread can use)\n");
                break;
            case cudaComputeModeProhibited:
                printf("Prohibited (no threads can use)\n");
                break;
            case cudaComputeModeExclusiveProcess:
                printf("Exclusive Process\n");
                break;
            default:
                printf("Unknown\n");
        }
        
        printf("\n");
    }
    
    // Runtime API version
    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Runtime Version: %d.%d\n", 
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    
    // Driver API version
    int driverVersion;
    cudaDriverGetVersion(&driverVersion);
    printf("CUDA Driver Version: %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10);
    
    printf("\n========================================\n");
    printf("   Device query completed successfully!\n");
    printf("========================================\n");
    
    return 0;
}
