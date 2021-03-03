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
#ifndef SOFA_HELPER_CUDAMEMORYMANAGER_H
#define SOFA_HELPER_CUDAMEMORYMANAGER_H

#include <sofa/helper/MemoryManager.h>
#include <cstring>
#include "mycuda.h"

#ifdef SOFACUDA_HAVE_SOFA_GL
#include <sofa/helper/system/gl.h>
#endif // SOFACUDA_HAVE_SOFA_GL

namespace sofa
{

namespace gpu
{

namespace cuda
{

//CPU MemoryManager
template <class T >
class CudaMemoryManager : public sofa::helper::MemoryManager<T>
{

public :
    template<class T2> struct rebind
    {
        typedef CudaMemoryManager<T2> other;
    };

    typedef T* host_pointer;
    typedef /*mutable*/ void* device_pointer;
#ifdef SOFACUDA_HAVE_SOFA_GL
    typedef GLuint gl_buffer;
#endif // SOFACUDA_HAVE_SOFA_GL

    enum { MAX_DEVICES = 8 };
    enum { BSIZE = 64 };
    enum { SUPPORT_GL_BUFFER = 1 };

    static int numDevices()
    {
        return mycudaGetnumDevices();
    }

    static void hostAlloc(void ** hPointer,int n)
    {
        mycudaMallocHost(hPointer,n);
    }

    static void hostFree(const host_pointer hSrcPointer)
    {
        mycudaFreeHost(hSrcPointer);
    }

    static void deviceAlloc(int d,void ** dPointer, int n)
    {
        mycudaMalloc(dPointer,n,d);
    }

    static void deviceFree(int d,const device_pointer dSrcPointer)
    {
        mycudaFree(dSrcPointer,d);
    }

    static void memsetHost(host_pointer hPointer, int value,size_t n) { memset((void*) hPointer, value, n); }

    static void memcpyHostToDevice(int d, device_pointer dDestPointer, const host_pointer hSrcPointer, size_t n)
    {
        if (mycudaVerboseLevel>=LOG_TRACE) std::cout << "CUDA: CPU->GPU copy of "<<sofa::core::objectmodel::BaseClass::decodeTypeName ( typeid ( *hSrcPointer ) ) <<": "<<n*sizeof(T) <<" B"<<std::endl;
        mycudaMemcpyHostToDevice(dDestPointer,hSrcPointer,n,d);
    }

    static void memcpyDeviceToHost(int d, host_pointer hDestPointer, const void * dSrcPointer , size_t n)
    {
        if (mycudaVerboseLevel>=LOG_TRACE) std::cout << "CUDA: GPU->CPU copy of "<<sofa::core::objectmodel::BaseClass::decodeTypeName ( typeid ( *hDestPointer ) ) <<": "<<n*sizeof(T) <<" B"<<std::endl;
        mycudaMemcpyDeviceToHost(hDestPointer,dSrcPointer,n,d);
    }

    static void memcpyDeviceToDevice(int d, device_pointer dDestPointer, const device_pointer dSrcPointer , size_t n)
    {
        mycudaMemcpyDeviceToDevice(dDestPointer,dSrcPointer,n,d);
    }

    static void memsetDevice(int d, device_pointer dDestPointer, int value,size_t n)
    {
        mycudaMemset(dDestPointer,value,n,d);
    }

    static int getBufferDevice()
    {
        return mycudaGetBufferDevice();
    }

    static bool bufferAlloc(gl_buffer* bId, int n, bool createBuffer = true)
    {
#ifdef SOFACUDA_HAVE_SOFA_GL
        if (n > 0)
        {
            if(createBuffer)
                glGenBuffers(1, bId);

            glBindBuffer( GL_ARRAY_BUFFER, *bId);
            glBufferData( GL_ARRAY_BUFFER, n, 0, GL_DYNAMIC_DRAW);
            glBindBuffer( GL_ARRAY_BUFFER, 0);
            return true;
        }
#endif // SOFACUDA_HAVE_SOFA_GL
        return false;
    }

    static void bufferFree(const gl_buffer bId)
    {
#ifdef SOFACUDA_HAVE_SOFA_GL
        glDeleteBuffers( 1, &bId);
#endif // SOFACUDA_HAVE_SOFA_GL
    }

    static bool bufferRegister(const gl_buffer bId)
    {
#ifdef SOFACUDA_HAVE_SOFA_GL
        mycudaGLRegisterBufferObject(bId);
#endif // SOFACUDA_HAVE_SOFA_GL
        return true;
    }

    static void bufferUnregister(const gl_buffer bId)
    {
#ifdef SOFACUDA_HAVE_SOFA_GL
        mycudaGLUnregisterBufferObject(bId);
#endif // SOFACUDA_HAVE_SOFA_GL
    }

    static bool bufferMapToDevice(device_pointer * dDestPointer, const gl_buffer bSrcId)
    {
#ifdef SOFACUDA_HAVE_SOFA_GL
        mycudaGLMapBufferObject(dDestPointer, bSrcId);
        return true;
#else
        return false;
#endif // SOFACUDA_HAVE_SOFA_GL
    }

    static void bufferUnmapToDevice(device_pointer * /*dDestPointer*/, const gl_buffer bSrcId)
    {
#ifdef SOFACUDA_HAVE_SOFA_GL
        mycudaGLUnmapBufferObject(bSrcId);
#endif // SOFACUDA_HAVE_SOFA_GL
    }

    static device_pointer deviceOffset(device_pointer dPointer,size_t offset)
    {
        return (T*)dPointer+offset;
    }

    static device_pointer null() {return NULL;}
    static bool isNull(device_pointer p) {return p==NULL;}
};

}

}

}

#endif //SOFA_HELPER_MEMORYMANAGER_H


