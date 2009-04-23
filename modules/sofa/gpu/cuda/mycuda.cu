/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "mycuda.h"
#include <cuda.h>
#ifdef WIN32
#include <sofa/helper/system/gl.h>
#endif
#include <cuda_gl_interop.h>

#include <sofa/helper/BackTrace.h>

//#define NO_CUDA

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

#ifdef NO_CUDA

bool cudaCheck(cudaError_t, const char*)
{
    return true;
}

bool cudaInitCalled = false;

int mycudaInit(int)
{
    cudaInitCalled = true;
    return 0;
}

void mycudaMalloc(void **devPtr, size_t)
{
    *devPtr = NULL;
}

void mycudaMallocPitch(void **devPtr, size_t*, size_t, size_t)
{
    *devPtr = NULL;
}

void mycudaFree(void *)
{
}

void mycudaMallocHost(void **hostPtr, size_t size)
{
    *hostPtr = malloc(size);
}

void mycudaFreeHost(void *hostPtr)
{
    free(hostPtr);
}

void mycudaMemcpyHostToDevice(void *, const void *, size_t)
{
}

void mycudaMemcpyDeviceToDevice(void *, const void *, size_t)
{
}

void mycudaMemcpyDeviceToHost(void *, const void *, size_t)
{
}

void mycudaMemcpyHostToDevice2D(void *, size_t, const void *, size_t, size_t, size_t)
{
}

void mycudaMemcpyDeviceToDevice2D(void *, size_t, const void *, size_t, size_t, size_t )
{
}

void mycudaMemcpyDeviceToHost2D(void *, size_t, const void *, size_t, size_t, size_t)
{
}

void mycudaGLRegisterBufferObject(int)
{
}

void mycudaGLUnregisterBufferObject(int)
{
}

void mycudaGLMapBufferObject(void** ptr, int)
{
    *ptr = NULL;
}

void mycudaGLUnmapBufferObject(int)
{
}

#else

bool cudaCheck(cudaError_t err, const char* src="?")
{
    if (err == cudaSuccess) return true;
    //fprintf(stderr, "CUDA: Error %d returned from %s.\n",(int)err,src);
    mycudaLogError(cudaGetErrorString(err), src);
    sofa::helper::BackTrace::dump();
    return false;
}

bool cudaInitCalled = false;

int mycudaInit(int device)
{
    int deviceCount = 0;
    cudaInitCalled = true;
    {
        const char* var = mygetenv("CUDA_MULTIOPS");
        if (var && *var)
        {
            mycudaMultiOpMax = atoi(var);
            if (mycudaMultiOpMax)
                myprintf("CUDA: Merging of up to %d identical operations enabled.\n", mycudaMultiOpMax);
            else
                myprintf("CUDA: Merging of identical operations disabled.\n", mycudaMultiOpMax);
        }
    }
    cudaCheck(cudaGetDeviceCount(&deviceCount),"cudaGetDeviceCount");
    myprintf("CUDA: %d device(s) found.\n", deviceCount);
    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp dev
#ifdef cudaDevicePropDontCare
            = cudaDevicePropDontCare
#endif
                    ;
        //memset(&dev,0,sizeof(dev));
        //dev.name=NULL;
        //dev.bytes=0;
        //dev.major=0;
        //dev.minor=0;
        cudaCheck(cudaGetDeviceProperties(&dev,i),"cudaGetDeviceProperties");
#if CUDA_VERSION >= 2010
        myprintf("CUDA:  %d : \"%s\", %d MB, %d cores at %.3f GHz, revision %d.%d",i,dev.name, dev.totalGlobalMem/(1024*1024), dev.multiProcessorCount*8, dev.clockRate * 1e-6f, dev.major, dev.minor);
        if (dev.kernelExecTimeoutEnabled)
            myprintf(", timeout enabled", dev.kernelExecTimeoutEnabled);
        myprintf("\n");
#elif CUDA_VERSION >= 2000
        myprintf("CUDA:  %d : \"%s\", %d MB, %d cores at %.3f GHz, revision %d.%d\n",i,dev.name, dev.totalGlobalMem/(1024*1024), dev.multiProcessorCount*8, dev.clockRate * 1e-6f, dev.major, dev.minor);
#else //if CUDA_VERSION >= 1000
        myprintf("CUDA:  %d : \"%s\", %d MB, cores at %.3f GHz, revision %d.%d\n",i,dev.name, dev.totalGlobalMem/(1024*1024), dev.clockRate * 1e-6f, dev.major, dev.minor);
//#else
//		myprintf("CUDA:  %d : \"%s\", %d MB, revision %d.%d\n",i,(dev.name==NULL?"":dev.name), dev.bytes/(1024*1024), dev.major, dev.minor);
#endif
    }
    if (device==-1)
    {
        const char* var = mygetenv("CUDA_DEVICE");
        device = (var && *var) ? atoi(var):0;
    }
    if (device >= deviceCount)
    {
        myprintf("CUDA: Device %d not found.\n", device);
        return 0;
    }
    else
    {
        cudaDeviceProp dev;
        cudaCheck(cudaGetDeviceProperties(&dev,device));
        myprintf("CUDA: Using device %d : \"%s\"\n",device,dev.name);
        cudaCheck(cudaSetDevice(device));
        return 1;
    }

}

void mycudaMalloc(void **devPtr, size_t size)
{
    if (!cudaInitCalled) mycudaInit();
    if (mycudaVerboseLevel>=LOG_INFO) myprintf("CUDA: malloc(%d).\n",size);
    cudaCheck(cudaMalloc(devPtr, size),"cudaMalloc");
    if (mycudaVerboseLevel>=LOG_TRACE) myprintf("CUDA: malloc(%d) -> 0x%x.\n",size, *devPtr);
}

void mycudaMallocPitch(void **devPtr, size_t* pitch, size_t width, size_t height)
{
    if (!cudaInitCalled) mycudaInit();
    if (mycudaVerboseLevel>=LOG_INFO) myprintf("CUDA: mallocPitch(%d,%d).\n",width,height);
    cudaCheck(cudaMallocPitch(devPtr, pitch, width, height),"cudaMalloc2D");
    if (mycudaVerboseLevel>=LOG_TRACE) myprintf("CUDA: mallocPitch(%d,%d) -> 0x%x at pitch %d.\n",width,height, *devPtr, (int)*pitch);
}

void mycudaFree(void *devPtr)
{
    if (mycudaVerboseLevel>=LOG_TRACE) myprintf("CUDA: free(0x%x).\n",devPtr);
    cudaCheck(cudaFree(devPtr),"cudaFree");
}

void mycudaMallocHost(void **hostPtr, size_t size)
{
    if (!cudaInitCalled) mycudaInit();
    if (mycudaVerboseLevel>=LOG_TRACE) myprintf("CUDA: mallocHost(%d).\n",size);
    cudaCheck(cudaMallocHost(hostPtr, size),"cudaMallocHost");
    if (mycudaVerboseLevel>=LOG_TRACE) myprintf("CUDA: mallocHost(%d) -> 0x%x.\n",size, *hostPtr);
}

void mycudaFreeHost(void *hostPtr)
{
    if (mycudaVerboseLevel>=LOG_TRACE) myprintf("CUDA: freeHost(0x%x).\n",hostPtr);
    cudaCheck(cudaFreeHost(hostPtr),"cudaFreeHost");
}

void mycudaMemcpyHostToDevice(void *dst, const void *src, size_t count)
{
    if (!cudaCheck(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice),"cudaMemcpyHostToDevice"))
        myprintf("in mycudaMemcpyHostToDevice(0x%x, 0x%x, %d)\n",dst,src,count);
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
    if (!cudaInitCalled) mycudaInit();
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

#endif

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
