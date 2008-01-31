#include "mycuda.h"
#include <cuda_gl_interop.h>
#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

void cudaCheck(cudaError_t err, const char* src="?")
{
    if (err == cudaSuccess) return;
    //fprintf(stderr, "CUDA: Error %d returned from %s.\n",(int)err,src);
    mycudaLogError(err, src);
}

bool cudaInitCalled = false;

int mycudaInit(int device)
{
    int deviceCount = 0;
    cudaInitCalled = true;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    myprintf("CUDA: %d devices found.\n", deviceCount);
    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp dev;
        memset(&dev,0,sizeof(dev));
        //dev.name=NULL;
        //dev.bytes=0;
        //dev.major=0;
        //dev.minor=0;
        cudaCheck(cudaGetDeviceProperties(&dev,i));
        //myprintf("CUDA:  %d : \"%s\", %d MB, revision %d.%d\n",i,(dev.name==NULL?"":dev.name), dev.bytes/(1024*1024), dev.major, dev.minor);
        myprintf("CUDA:  %d : \"%s\", %d MB, revision %d.%d\n",i,dev.name, dev.totalGlobalMem/(1024*1024), dev.major, dev.minor);
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
    if (!cudaInitCalled) mycudaInit(0);
    myprintf("CUDA: malloc(%d).\n",size);
    cudaCheck(cudaMalloc(devPtr, size),"cudaMalloc");
}

void mycudaMallocPitch(void **devPtr, size_t* pitch, size_t width, size_t height)
{
    if (!cudaInitCalled) mycudaInit(0);
    myprintf("CUDA: mallocPitch(%d,%d).\n",width,height);
    cudaCheck(cudaMallocPitch(devPtr, pitch, width, height),"cudaMalloc2D");
    myprintf("pitch=%d\n",*pitch);
}

void mycudaFree(void *devPtr)
{
    myprintf("CUDA: free().\n");
    cudaCheck(cudaFree(devPtr),"cudaFree");
}

void mycudaMallocHost(void **hostPtr, size_t size)
{
    if (!cudaInitCalled) mycudaInit(0);
    myprintf("CUDA: mallocHost(%d).\n",size);
    cudaCheck(cudaMallocHost(hostPtr, size),"cudaMallocHost");
}

void mycudaFreeHost(void *hostPtr)
{
    myprintf("CUDA: freeHost().\n");
    cudaCheck(cudaFreeHost(hostPtr),"cudaFreeHost");
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

void mycudaMemcpyHostToDevice2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height)
{
    cudaCheck(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice),"cudaMemcpyHostToDevice2D");
}

void mycudaMemcpyDeviceToDevice2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height)
{
    cudaCheck(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice),"cudaMemcpyDeviceToDevice2D");
}

void mycudaMemcpyDeviceToHost2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height)
{
    cudaCheck(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost),"cudaMemcpyDeviceToHost2D");
}

void mycudaGLRegisterBufferObject(int id)
{
    if (!cudaInitCalled) mycudaInit(0);
    cudaCheck(cudaGLRegisterBufferObject((GLuint)id),"cudaGLRegisterBufferObject");
}

void mycudaGLUnregisterBufferObject(int id)
{
    cudaCheck(cudaGLUnregisterBufferObject((GLuint)id),"cudaGLUnregisterBufferObject");
}

void mycudaGLMapBufferObject(void** ptr, int id)
{
    cudaCheck(cudaGLMapBufferObject(ptr, (GLuint)id),"cudaGLMapBufferObject");
}

void mycudaGLUnmapBufferObject(int id)
{
    cudaCheck(cudaGLUnmapBufferObject((GLuint)id),"cudaGLUnmapBufferObject");
}

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
