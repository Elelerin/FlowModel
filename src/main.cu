#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <curand.h>
#include <thrust/host_vector.h>
#define DEBUG
__device__ int devData;

__global__ void helloFromGPU(){
        //printf("THREAD ID: %d\n", threadIdx.x);
        //printf("BLOCK ID: %d\n", blockDim.x * blockIdx.x + threadIdx.x);
        //printf("THREADS ON BLOCK: %d\n", blockDim.x);
        //printf("BLOCKS ON GRID:%d\n", gridDim.x);
        atomicAdd(&devData, 1);
        printf("devData @ thread id %d: %d\n", blockDim.x * blockIdx.x + threadIdx.x, devData);
}

int main(void)
{
    int f = 0;
    cudaMemcpyToSymbol(devData, &f, sizeof(int));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    short MPC = deviceProp.multiProcessorCount;
    #ifdef DEBUG
    printf("MULTI-PROCESSOR COUNT TO BE USED IN TESTING: %d", MPC);
    #endif

    helloFromGPU <<<32, 32>>>();


    cudaDeviceSynchronize();
    int outputData;
    cudaMemcpyFromSymbol(&outputData, devData, sizeof(int));
    if(outputData == 1024){
        printf("SUCCESSFUL RUN!");
    }else{
        printf("OUTPUT DATA IS: %d!\n", outputData);
    }

    cudaDeviceReset();
    return 0;
}
