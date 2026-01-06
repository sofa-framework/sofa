/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifdef _WIN32
#include <windows.h>
#endif

#include "mycuda.h"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <iostream>

cudaDeviceProp mycudaDeviceProp;


#if defined(__cplusplus)

#define STRINGIFY(x) #x
#define _STR(x) STRINGIFY(x)
SOFA_PRAGMA_MESSAGE("__cplusplus value: " _STR(__cplusplus))

namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C"
{
    int SOFA_GPU_CUDA_API mycudaGetMultiProcessorCount();
    void cuda_void_kernel();
}

bool cudaCheck(cudaError_t err, const char* src="?")
{
    if (err == cudaSuccess) return true;
    //fprintf(stderr, "CUDA: Error %d returned from %s.\n",(int)err,src);
    mycudaLogError(cudaGetErrorString(err), src);
    return false;
}

bool cudaInitCalled = false;
int deviceCount = 0;

#ifdef SOFA_WITH_DEVTOOLS
__global__ void print_cuda_standard()
{
    /**
    * 199711L = C++98
    * 201103L = C++11
    * 201402L = C++14
    * 201703L = C++17
    * 202002L = C++20
    */
    printf("CUDA Standard: %ld\n", __cplusplus);
}
#endif

int mycudaInit(int device)
{
    if (cudaInitCalled) return 1;

#if defined(__cplusplus)
    mycudaPrintf("C++ standard = %ld", __cplusplus);
#endif

    cudaInitCalled = true;
    const cudaError_t getDeviceCountError = cudaGetDeviceCount(&deviceCount);
    if (getDeviceCountError != cudaSuccess)
    {
        mycudaPrintfError("error returned from cudaGetDeviceCount: %s", cudaGetErrorString(getDeviceCountError));
        return 0;
    }
    mycudaPrintf("CUDA: %d device(s) found.\n", deviceCount);
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

        size_t free, total;
        cudaMemGetInfo(&free,&total);

#if CUDA_VERSION >= 2010
    #if CUDA_VERSION < 13000
        mycudaPrintf("CUDA:  %d : \"%s\", %d/%d MB, %d cores at %.3f GHz, revision %d.%d",i,dev.name, free/(1024*1024), dev.totalGlobalMem/(1024*1024), dev.multiProcessorCount*8, dev.clockRate * 1e-6f, dev.major, dev.minor);
        if (dev.kernelExecTimeoutEnabled)
            mycudaPrintf(", timeout enabled");
        mycudaPrintf("\n");
    #else
        mycudaPrintf("CUDA:  %d : \"%s\", %d/%d MB, %d cores, revision %d.%d\n",i,dev.name, free/(1024*1024), dev.totalGlobalMem/(1024*1024), dev.multiProcessorCount*8, dev.major, dev.minor);
    #endif
#elif CUDA_VERSION >= 2000
    #if CUDA_VERSION < 13000
        mycudaPrintf("CUDA:  %d : \"%s\", %d/%d MB, %d cores at %.3f GHz, revision %d.%d\n",i,dev.name, free/(1024*1024), dev.totalGlobalMem/(1024*1024), dev.multiProcessorCount*8, dev.clockRate * 1e-6f, dev.major, dev.minor);
    #else
        mycudaPrintf("CUDA:  %d : \"%s\", %d/%d MB, %d cores, revision %d.%d\n",i,dev.name, free/(1024*1024), dev.totalGlobalMem/(1024*1024), dev.multiProcessorCount*8, dev.major, dev.minor);
    #endif
#else //if CUDA_VERSION >= 1000
    #if CUDA_VERSION < 13000
        mycudaPrintf("CUDA:  %d : \"%s\", %d/%d MB, cores at %.3f GHz, revision %d.%d\n",i,dev.name, free/(1024*1024), dev.totalGlobalMem/(1024*1024), dev.clockRate * 1e-6f, dev.major, dev.minor);
    #else
        mycudaPrintf("CUDA:  %d : \"%s\", %d/%d MB, cores, revision %d.%d\n",i,dev.name, free/(1024*1024), dev.totalGlobalMem/(1024*1024), dev.major, dev.minor);
    #endif
#endif
    }
    if (device==-1)
    {
        const char* var = mygetenv("CUDA_DEVICE");
        device = (var && *var) ? atoi(var):0;
    }
    if (device >= deviceCount)
    {
        mycudaPrintf("CUDA: Device %d not found.\n", device);
        return 0;
    }
    else
    {
        cudaDeviceProp& dev = mycudaDeviceProp;
        cudaCheck(cudaGetDeviceProperties(&dev,device));
        mycudaPrintf("CUDA: Using device %d : \"%s\"\n",device,dev.name);
        cudaCheck(cudaSetDevice(device));
        mycudaPrivateInit(device);
    }


#if defined(SOFA_GPU_CUBLAS) && !defined(SOFA_GPU_CUBLAS_V2)
    cublasInit();
#endif

#ifdef SOFA_WITH_DEVTOOLS
    print_cuda_standard<<<1, 1>>>();
    cudaCheck(cudaDeviceSynchronize());
#endif

    return 1;
}

int mycudaGetMultiProcessorCount()
{
    return mycudaDeviceProp.multiProcessorCount;
}

void mycudaMalloc(void **devPtr, size_t size,int /*d*/)
{
    if (!cudaInitCalled) mycudaInit();
    if (mycudaVerboseLevel>=LOG_INFO) mycudaPrintf("CUDA: malloc(%d).\n",size);
    cudaCheck(cudaMalloc(devPtr, size),"cudaMalloc");
    if (mycudaVerboseLevel>=LOG_TRACE) mycudaPrintf("CUDA: malloc(%d) -> 0x%x.\n",size, *devPtr);
}

void mycudaMallocPitch(void **devPtr, size_t* pitch, size_t width, size_t height)
{
    if (!cudaInitCalled) mycudaInit();
    if (mycudaVerboseLevel>=LOG_INFO) mycudaPrintf("CUDA: mallocPitch(%d,%d).\n",width,height);
    cudaCheck(cudaMallocPitch(devPtr, pitch, width, height),"cudaMalloc2D");
    if (mycudaVerboseLevel>=LOG_TRACE) mycudaPrintf("CUDA: mallocPitch(%d,%d) -> 0x%x at pitch %d.\n",width,height, *devPtr, (int)*pitch);
}

void mycudaFree(void *devPtr,int /*d*/)
{
    if (mycudaVerboseLevel>=LOG_TRACE) mycudaPrintf("CUDA: free(0x%x).\n",devPtr);
    cudaCheck(cudaFree(devPtr),"cudaFree");
}

void mycudaMallocHost(void **hostPtr, size_t size)
{
    if (!cudaInitCalled) mycudaInit();
    if (mycudaVerboseLevel>=LOG_TRACE) mycudaPrintf("CUDA: mallocHost(%d).\n",size);
    cudaCheck(cudaMallocHost(hostPtr, size),"cudaMallocHost");
    if (mycudaVerboseLevel>=LOG_TRACE) mycudaPrintf("CUDA: mallocHost(%d) -> 0x%x.\n",size, *hostPtr);
}

void mycudaFreeHost(void *hostPtr)
{
    if (mycudaVerboseLevel>=LOG_TRACE) mycudaPrintf("CUDA: freeHost(0x%x).\n",hostPtr);
    cudaCheck(cudaFreeHost(hostPtr),"cudaFreeHost");
}

void mycudaMemcpyHostToDevice(void *dst, const void *src, size_t count,int /*d*/)
{
    //count = (count+3)&(size_t)-4;
    if (!cudaCheck(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice),"cudaMemcpyHostToDevice"))
        mycudaPrintf("in mycudaMemcpyHostToDevice(0x%x, 0x%x, %d)\n",dst,src,count);

    if (mycudaVerboseLevel>=LOG_STACK_TRACE) displayStack("mycudaMemcpyHostToDevice");
}

void mycudaMemcpyDeviceToDevice(void *dst, const void *src, size_t count,int /*d*/		)
{
    cudaCheck(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice),"cudaMemcpyDeviceToDevice");
}

void mycudaMemcpyDeviceToHost(void *dst, const void *src, size_t count,int /*d*/)
{
    //count = (count+3)&(size_t)-4;
    cudaCheck(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost),"cudaMemcpyDeviceToHost");

    if (mycudaVerboseLevel>=LOG_STACK_TRACE) displayStack("mycudaMemcpyDeviceToHost");
}

void mycudaMemcpyHostToDevice2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height)
{
    cudaCheck(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice),"cudaMemcpyHostToDevice2D");

    if (mycudaVerboseLevel>=LOG_STACK_TRACE) displayStack("mycudaMemcpyHostToDevice2D");
}

void mycudaMemcpyDeviceToDevice2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height)
{
    cudaCheck(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice),"cudaMemcpyDeviceToDevice2D");
}

void mycudaMemcpyDeviceToHost2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height)
{
    cudaCheck(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost),"cudaMemcpyDeviceToHost2D");

    if (mycudaVerboseLevel>=LOG_STACK_TRACE) displayStack("mycudaMemcpyDeviceToHost2D");
}

void mycudaMemset(void *devPtr, int val, size_t size,int d)
{
    cudaCheck(cudaMemset(devPtr, val,size),"mycudaMemset");
}



void mycudaThreadSynchronize()
{
    if (!cudaInitCalled) return; // no need to synchronize if no-one used cuda yet

    cudaDeviceSynchronize();
}

#if CUDA_VERSION >= 4000

void mycudaDeviceSynchronize()
{
	if (!cudaInitCalled) return;
	
	cudaDeviceSynchronize();
}
#endif

void mycudaCheckError(const char* src)
{
    if (!cudaInitCalled) return; // no need to check errors if no-one used cuda yet
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError(),src);
}

void mycudaGLRegisterBufferObject(int id)
{
    if (!cudaInitCalled) mycudaInit();
    mycudaPrintf("mycudaGLRegisterBufferObject %d\n",id);
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

int mycudaGetnumDevices()
{
    if (!cudaInitCalled) mycudaInit();
    return deviceCount;
}

int mycudaGetBufferDevice()
{
    return 0;
}

__global__ void cuda_debug_kernel()
{
}

void cuda_void_kernel()
{
    mycudaPrintf("WARNING : cuda_void_kernel should only be used for debug\n");

    dim3 threads(1,1);
    dim3 grid(1,1);
    {cuda_debug_kernel<<< grid, threads >>>(); mycudaDebugError("cuda_debug_kernel");}
}

#ifdef SOFA_GPU_CUBLAS

cublasHandle_t getCublasCtx()
{
    static cublasHandle_t cublashandle = NULL;
    if (cublashandle==NULL)
    {
        cublasStatus_t status = cublasCreate(&cublashandle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            mycudaPrintf("cublas Handle init failed\n");
        }
    }
    return cublashandle;
}

cusparseHandle_t getCusparseCtx()
{
    static cusparseHandle_t cusparsehandle = NULL;
    if (cusparsehandle==NULL)
    {
        cusparseStatus_t status = cusparseCreate(&cusparsehandle);
        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            mycudaPrintf("cusparse Handle init failed\n");
        }
    }
    return cusparsehandle;
}

static cusparseMatDescr_t matdescGen=NULL;

cusparseMatDescr_t getCusparseMatGeneralDescr()
{
    if (matdescGen==NULL)
    {
        cusparseStatus_t status = cusparseCreateMatDescr(&matdescGen);
        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            mycudaPrintf("Matrix descriptor init failed\n");
        }
        cusparseSetMatIndexBase(matdescGen, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(matdescGen, CUSPARSE_MATRIX_TYPE_GENERAL);
    }

    return matdescGen;
}

static cusparseMatDescr_t matdescTriLower=NULL;

cusparseMatDescr_t getCusparseMatTriangularLowerDescr()
{
    if (matdescTriLower==NULL)
    {
        cusparseStatus_t status = cusparseCreateMatDescr(&matdescTriLower);
        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            mycudaPrintf("Matrix descriptor init failed\n");
        }
        cusparseSetMatType ( matdescTriLower, CUSPARSE_MATRIX_TYPE_TRIANGULAR );
        cusparseSetMatIndexBase ( matdescTriLower, CUSPARSE_INDEX_BASE_ZERO );
        cusparseSetMatDiagType ( matdescTriLower, CUSPARSE_DIAG_TYPE_UNIT );
        cusparseSetMatFillMode ( matdescTriLower, CUSPARSE_FILL_MODE_LOWER );
    }

    return matdescTriLower;
}

static cusparseMatDescr_t matdescTriUpper=NULL;

cusparseMatDescr_t getCusparseMatTriangularUpperDescr()
{
    if (matdescTriUpper==NULL)
    {
        cusparseStatus_t status = cusparseCreateMatDescr(&matdescTriUpper);
        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            mycudaPrintf("Matrix descriptor init failed\n");
        }
        cusparseSetMatType ( matdescTriUpper, CUSPARSE_MATRIX_TYPE_TRIANGULAR );
        cusparseSetMatIndexBase ( matdescTriUpper, CUSPARSE_INDEX_BASE_ZERO );
        cusparseSetMatDiagType ( matdescTriUpper, CUSPARSE_DIAG_TYPE_UNIT );
        cusparseSetMatFillMode ( matdescTriUpper, CUSPARSE_FILL_MODE_UPPER );
    }

    return matdescTriUpper;
}

void SOFA_GPU_CUDA_API mycudaMemGetInfo(size_t * free,size_t * total) {
    cudaMemGetInfo(free,total);
}

#endif //SOFA_GPU_CUBLAS

#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif
