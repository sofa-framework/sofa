#ifndef SOFA_HELPER_OPENCLMEMORYMANAGER_H
#define SOFA_HELPER_OPENCLMEMORYMANAGER_H

#include "other/MemoryManager.h"
#include "OpenCLManager.h"
#include "CL/cl.h"
#include "GL/gl.h"
#include <iostream>
#include "stdlib.h"

using namespace std;


namespace sofa
{

namespace helper
{

/**
  * OpenCLMemoryManager
  * OpenCL and multi-GPU version of MemoryManager
  */
template <class T>
class OpenCLMemoryManager: public MemoryManager<T>
{
public:
    typedef T* host_pointer;
    typedef void* device_pointer;

    enum { MAX_DEVICES = 8 };
    enum { BSIZE = 32 };
    enum { SUPPORT_GL_BUFFER = 0 };

private:


public:


    /**
      * Return devices counting
      */
    static int numDevices()
    {
        return OpenCLManager::numDevices();
    }

    /**
      * alloc the device memory
      * @param n size of memory
      */
    static device_pointer deviceAlloc(int, int n)
    {
        cl_mem* mem = new cl_mem();
        *mem = clCreateBuffer(OpenCLManager::context(),CL_MEM_READ_WRITE,n,NULL,&OpenCLManager::error());
        return (device_pointer) mem;
    }

    /**
      * free the memory device
      * @param dSrcPointer memory pointer to free
      */
    static device_pointer deviceFree(const device_pointer dSrcPointer)
    {
        OpenCLManager::error() = clReleaseMemObject(*((cl_mem*)dSrcPointer));
        return dSrcPointer;
    }

    /**
      * copy memory from Host to Device
      * @param d device number
      * @param dDestPointer device memory pointer (destination)
      * @param hSrcPointer host memory pointer (source)
      * @param n size of data
      */
    static  void memcpyHostToDevice(int d, device_pointer dDestPointer, const host_pointer hSrcPointer, size_t n)
    {
        OpenCLManager::error() = clEnqueueWriteBuffer(OpenCLManager::queue(d), *((cl_mem*)dDestPointer), CL_TRUE, 0, n, hSrcPointer,0,NULL,NULL);
    }

    /**
      * copy memory from Host to Device
      * @param d device number
      * @param hDestPointer host memory pointer (destination)
      * @param dSrcPointer device memory pointer (source)
      * @param n size of data
      */
    static  void memcpyDeviceToHost(int d, host_pointer hDestPointer, const device_pointer dSrcPointer , size_t n)
    {
        OpenCLManager::error() = clEnqueueReadBuffer(OpenCLManager::queue(d), *((cl_mem*)dSrcPointer), CL_TRUE, 0, n,hDestPointer,0,NULL,NULL);
    }

    /**
      * copy memory from Device to Device
      * @param dDest device number (destination)
      * @param dSrc device number (source)
      * @param dDestPointer device memory pointer (destination)
      * @param dSrcPointer device memory pointer (source)
      * @param n size of data
      */
    static  void memcpyDeviceToDevice(int dDest, int dSrc, device_pointer dDestPointer, const device_pointer dSrcPointer , size_t n)
    {
        OpenCLManager::error() = clEnqueueCopyBuffer(OpenCLManager::queue(dDest),*((cl_mem*)dDestPointer),*((cl_mem*)dSrcPointer),0,0, n,0,NULL,NULL);
    }

    /**
      * set the values of array
      * @param  d device number
      * @param dDestPointer device memory pointer
      * @param value value to set
      * @param n size of data
      */
    static  void memsetDevice(int d, device_pointer dDestPointer, int value,size_t n)
    {
        device_pointer array = new T[n];
        memset((void*) array, value, n);
        clEnqueueWriteBuffer(OpenCLManager::queue(d), *((cl_mem*)dDestPointer), CL_TRUE, 0,n, array,0,NULL,NULL);
    }


    static void hostAlloc(void ** hPointer,int n) { *hPointer = (void *)malloc(n); }
    static void memsetHost(host_pointer hPointer, int value,size_t n) { memset((void*) hPointer, value, n); }
    static device_pointer hostFree(const host_pointer hSrcPointer) {free(hSrcPointer); return hSrcPointer;}


    /*
    	static bool bufferAlloc(gl_buffer* bId, int n) { return false; }
    	static void bufferFree(const gl_buffer bId) {}

    	static bool bufferRegister(const gl_buffer bId) {}
    	static void bufferUnregister(const gl_buffer bId) {}

    	static bool bufferMapToDevice(device_pointer* dDestPointer, const gl_buffer bSrcId)
    	{
    		if(*dDestPointer)clReleaseMemObject(*((cl_mem*)dDestPointer));
    		clCreateFromGLBuffer(_context, CL_MEM_READ_WRITE, bSrcId, &error);
    	}

    	static void bufferUnmapToDevice(device_pointer dDestPointer, const gl_buffer bSrcId) {}*/





};

}

}



#include "OpenCLMemoryManager.inl"
#endif
