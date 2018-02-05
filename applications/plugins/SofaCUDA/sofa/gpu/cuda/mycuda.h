/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef MYCUDA_H
#define MYCUDA_H

#include <SofaCUDA/config.h>
#include <string.h>

#ifdef SOFA_GPU_CUBLAS
#include <cublas.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#endif

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C" {

    extern SOFA_GPU_CUDA_API int  mycudaGetnumDevices();
    extern SOFA_GPU_CUDA_API int  mycudaGetBufferDevice();

    extern int SOFA_GPU_CUDA_API mycudaInit(int device=-1);
    extern void SOFA_GPU_CUDA_API mycudaMalloc(void **devPtr, size_t size,int d = mycudaGetBufferDevice());
    extern void SOFA_GPU_CUDA_API mycudaMallocPitch(void **devPtr, size_t* pitch, size_t width, size_t height);
    extern void SOFA_GPU_CUDA_API mycudaFree(void *devPtr,int d = mycudaGetBufferDevice());
    extern void SOFA_GPU_CUDA_API mycudaMallocHost(void **hostPtr, size_t size);
    extern void SOFA_GPU_CUDA_API mycudaFreeHost(void *hostPtr);
//extern void SOFA_GPU_CUDA_API mycudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    extern void SOFA_GPU_CUDA_API mycudaMemcpyHostToDevice(void *dst, const void *src, size_t count,int d = mycudaGetBufferDevice());
    extern void SOFA_GPU_CUDA_API mycudaMemcpyDeviceToDevice(void *dst, const void *src, size_t count,int d = mycudaGetBufferDevice());
    extern void SOFA_GPU_CUDA_API mycudaMemcpyDeviceToHost(void *dst, const void *src, size_t count,int d = mycudaGetBufferDevice());
    extern void SOFA_GPU_CUDA_API mycudaMemcpyHostToDevice2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
    extern void SOFA_GPU_CUDA_API mycudaMemcpyDeviceToDevice2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
    extern void SOFA_GPU_CUDA_API mycudaMemcpyDeviceToHost2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);

    extern void SOFA_GPU_CUDA_API mycudaGLRegisterBufferObject(int id);
    extern void SOFA_GPU_CUDA_API mycudaGLUnregisterBufferObject(int id);

    extern void SOFA_GPU_CUDA_API mycudaGLMapBufferObject(void** ptr, int id);
    extern void SOFA_GPU_CUDA_API mycudaGLUnmapBufferObject(int id);

    extern void SOFA_GPU_CUDA_API mycudaMemset(void * devPtr, int val , size_t size,int d = mycudaGetBufferDevice());

	/*CAUTION : DEPRECATED SINCE CUDA 4.0 */
    extern void SOFA_GPU_CUDA_API mycudaThreadSynchronize();

#if CUDA_VERSION >= 4000
	extern void SOFA_GPU_CUDA_API mycudaDeviceSynchronize();
#endif
	
	
    extern void SOFA_GPU_CUDA_API mycudaCheckError(const char* src);

    extern void SOFA_GPU_CUDA_API displayStack(const char * name);

    extern void SOFA_GPU_CUDA_API mycudaMemGetInfo(size_t *  	free,size_t *  	total);


#ifdef SOFA_GPU_CUBLAS
    extern cusparseHandle_t SOFA_GPU_CUDA_API getCusparseCtx();
    extern cublasHandle_t SOFA_GPU_CUDA_API getCublasCtx();
    extern cusparseMatDescr_t SOFA_GPU_CUDA_API getCusparseMatGeneralDescr();
    extern cusparseMatDescr_t SOFA_GPU_CUDA_API getCusparseMatTriangularUpperDescr();
    extern cusparseMatDescr_t SOFA_GPU_CUDA_API getCusparseMatTriangularLowerDescr();

#endif

#if defined(NDEBUG) && !defined(CUDA_DEBUG)
#define mycudaDebugError(src) do {} while(0)
#else
#define mycudaDebugError(src) ::sofa::gpu::cuda::mycudaCheckError(src)
#endif

// To add a call to mycudaDebugError after all kernel launches in a file, you can use :
// sed -i.bak -e 's/\([ \t]\)\([_A-Za-z][_A-Za-z0-9]*[ \t]*\(<[_A-Za-z0-9 :,().+*\/|&^-]*\(<[_A-Za-z0-9 :,().+*\/|&^-]*>[_A-Za-z0-9 :,().+*\/|&^-]*\)*>\)\?[:space:]*\)<<</\1###\2###<<</g' -e's/###\([^;#]*[^;# \t]\)\([:space:]*\)###\(<<<[^;]*>>>[^;]*\);/{\1\2\3; mycudaDebugError("\1");}/g' -e 's/###\([^;#]*\)###\(<<<\)/\1\2/g' myfile.cu

    extern void SOFA_GPU_CUDA_API mycudaLogError(const char* err, const char* src);
    extern int SOFA_GPU_CUDA_API mycudaPrintf(const char* fmt, ...);
    extern int SOFA_GPU_CUDA_API mycudaGetMultiProcessorCount();
    extern void mycudaPrivateInit(int device=-1);

    extern void cuda_void_kernel();


    extern const char* mygetenv(const char* name);

    enum MycudaVerboseLevel
    {
        LOG_NONE = 0,
        LOG_ERR = 1,
        LOG_INFO = 2,
        LOG_TRACE = 3,
        LOG_STACK_TRACE = 4
    };

    extern MycudaVerboseLevel SOFA_GPU_CUDA_API mycudaVerboseLevel;
}


#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif

#endif
