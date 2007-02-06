#include "mycuda.h"

// \todo Why mycuda.h is not read correctly...
extern "C" {
    extern int mycudaInit(int device);
    extern void mycudaMalloc(void **devPtr, size_t size);
    extern void mycudaFree(void *devPtr);
//extern void mycudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    extern void mycudaMemcpyHostToDevice(void *dst, const void *src, size_t count);
    extern void mycudaMemcpyDeviceToDevice(void *dst, const void *src, size_t count);
    extern void mycudaMemcpyDeviceToHost(void *dst, const void *src, size_t count);

    extern void mycudaLogError(int err, const char* src);
    extern int myprintf(const char* fmt, ...);
}

void cudaCheck(cudaError_t err, const char* src="?")
{
    if (err == cudaSuccess) return;
    //fprintf(stderr, "CUDA: Error %d returned from %s.\n",(int)err,src);
    mycudaLogError(err, src);
}

int mycudaInit(int device)
{
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    myprintf("CUDA: %d devices found.\n", deviceCount);
    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp dev;
        dev.name=NULL;
        dev.mbytes=0;
        dev.major=0;
        dev.minor=0;
        cudaCheck(cudaGetDeviceProperties(&dev,i));
        myprintf("CUDA:  %d : \"%s\", %d MB, revision %d.%d\n",i,(dev.name==NULL?"":dev.name), dev.mbytes, dev.major, dev.minor);
    }
    if (device >= deviceCount)
    {
        myprintf("CUDA: Device %d not found.\n", device);
        return 0;
    }
    else
    {
        cudaCheck(cudaSetDevice(device));
        return 1;
    }
}

void mycudaMalloc(void **devPtr, size_t size)
{
    myprintf("CUDA: malloc(%d).\n",size);
    cudaCheck(cudaMalloc(devPtr, size),"cudaMalloc");
}

void mycudaFree(void *devPtr)
{
    myprintf("CUDA: free().\n");
    cudaCheck(cudaFree(devPtr),"cudaFree");
}

void mycudaMemcpyHostToDevice(void *dst, const void *src, size_t count)
{
    cudaCheck(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice),"cudaMemcpyHostToDevice");
}

void mycudaMemcpyDeviceToDevice(void *dst, const void *src, size_t count)
{
    cudaCheck(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice),"cudaMemcpyDeviceToDevice");
}

void mycudaMemcpyDeviceToHost(void *dst, const void *src, size_t count)
{
    cudaCheck(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost),"cudaMemcpyDeviceToHost");
}
