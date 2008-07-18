/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef MYCUDA_H
#define MYCUDA_H

#include <string.h>
#include <sofa/helper/system/gl.h>

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace cuda
{
#endif

extern "C" {
    extern int mycudaInit(int device=-1);
    extern void mycudaMalloc(void **devPtr, size_t size);
    extern void mycudaMallocPitch(void **devPtr, size_t* pitch, size_t width, size_t height);
    extern void mycudaFree(void *devPtr);
    extern void mycudaMallocHost(void **hostPtr, size_t size);
    extern void mycudaFreeHost(void *hostPtr);
//extern void mycudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    extern void mycudaMemcpyHostToDevice(void *dst, const void *src, size_t count);
    extern void mycudaMemcpyDeviceToDevice(void *dst, const void *src, size_t count);
    extern void mycudaMemcpyDeviceToHost(void *dst, const void *src, size_t count);
    extern void mycudaMemcpyHostToDevice2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
    extern void mycudaMemcpyDeviceToDevice2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
    extern void mycudaMemcpyDeviceToHost2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);

    extern void mycudaGLRegisterBufferObject(int id);
    extern void mycudaGLUnregisterBufferObject(int id);

    extern void mycudaGLMapBufferObject(void** ptr, int id);
    extern void mycudaGLUnmapBufferObject(int id);

    extern void mycudaLogError(int err, const char* src);
    extern int myprintf(const char* fmt, ...);
    extern const char* mygetenv(const char* name);

    enum MycudaVerboseLevel
    {
        LOG_NONE = 0,
        LOG_ERR = 1,
        LOG_INFO = 2,
        LOG_TRACE = 3
    };

    extern MycudaVerboseLevel mycudaVerboseLevel;
}


#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif

#endif
