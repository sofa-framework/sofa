#ifndef SOFA_GPU_OPENCLMEMORYMANAGER_H
#define SOFA_GPU_OPENCLMEMORYMANAGER_H

#include "myopencl.h"
#include <sofa/helper/MemoryManager.h>
#include <iostream>
#include <stdlib.h>

namespace sofa
{

namespace gpu
{

namespace opencl
{


void OpenCLMemoryManager_memsetDevice(int d, _device_pointer a, int value, size_t n);

/**
  * OpenCLMemoryManager
  * OpenCL and multi-GPU version of MemoryManager
  */
template <class T>
class OpenCLMemoryManager: public sofa::helper::MemoryManager<T>
{

public :
    typedef T* host_pointer;

    typedef _device_pointer device_pointer ;

    typedef GLuint gl_buffer;

    enum { MAX_DEVICES = 8 };
    enum { BSIZE = 32 };
    enum { SUPPORT_GL_BUFFER = 0 };

    static int numDevices()
    {
        return gpu::opencl::myopenclNumDevices();
    }

    static void hostAlloc(void ** hPointer,int n)
    {
        *hPointer = (host_pointer) malloc(n);
    }

    static void memsetHost(host_pointer hPointer, int value,size_t n)
    {

        memset((void*) hPointer, value, n);
    }

    static void hostFree(const host_pointer hSrcPointer)
    {
        free(hSrcPointer);
    }


    static void deviceAlloc(int d,device_pointer* dPointer, int n)
    {
        myopenclCreateBuffer(d,&(*dPointer).m,n);
        dPointer->offset=0;
        dPointer->_null=false;
    }

    static void deviceFree(int d,/*const*/ device_pointer dSrcPointer)
    {
        myopenclReleaseBuffer(d,dSrcPointer.m);
        dSrcPointer.offset=0;
        dSrcPointer._null=true;
    }

    static void memcpyHostToDevice(int d, device_pointer dDestPointer,const host_pointer hSrcPointer, size_t n)
    {
        myopenclEnqueueWriteBuffer(d,(dDestPointer).m,(dDestPointer).offset,hSrcPointer,n);
    }

    static void memcpyDeviceToHost(int d, host_pointer hDestPointer, const device_pointer dSrcPointer, size_t n)
    {
        myopenclEnqueueReadBuffer(d,hDestPointer,(dSrcPointer).m,(dSrcPointer).offset,n);
    }

    static void memcpyDeviceToDevice(int d, device_pointer dDestPointer, const device_pointer dSrcPointer , size_t n)
    {
        myopenclEnqueueCopyBuffer(d, (dDestPointer).m,(dDestPointer).offset, (dSrcPointer).m,(dSrcPointer).offset, n);
    }

    static void memsetDevice(int d, device_pointer dDestPointer, int value, size_t n)
    {
        OpenCLMemoryManager_memsetDevice(d, dDestPointer, value, n);
    }

//	static void memsetDevice(int d, device_pointer dDestPointer, int value, size_t n)
//	{
//		myopenclMemsetDevice(d,dDestPointer,value,n);
//	}



    /*	static void memsetDevice(int d, device_pointer dDestPointer, int value, size_t n)
    	{
    		T* array = new T[n];
    		memset((void*)array, value, n);
    		myopenclEnqueueWriteBuffer(d,(dDestPointer).m,(dDestPointer).offset,(void*)array,n);
    		delete(array);
    	}*/



    static int getBufferDevice()
    {
        return 0;
    }

    static bool bufferAlloc(gl_buffer*/* bId*/, int/* n*/)
    {
        return false;
    }

    static void bufferFree(const gl_buffer /*bId*/)
    {

    }

    static bool bufferRegister(const gl_buffer /*bId*/)
    {
        return false;
    }

    static void bufferUnregister(const gl_buffer /*bId*/)
    {

    }

    static bool bufferMapToDevice(void* dDestPointer, const gl_buffer /*bSrcId*/)
    {
        device_pointer* d=(device_pointer*)dDestPointer;
        d->m=d->m;		//delete this line when implementation
        return false;
    }

    static void bufferUnmapToDevice(void*  dDestPointer, const gl_buffer /*bSrcId*/)
    {
        device_pointer* d=(device_pointer*)dDestPointer;
        d->m=d->m;		//delete this line when implementation
    }

    static device_pointer deviceOffset(device_pointer memory,size_t offset)
    {
        device_pointer p;
        p.m = memory.m;
        p.offset = memory.offset + offset*sizeof(T);
        return p;
    }

    static device_pointer null()
    {
        return device_pointer();
    }

    static void null(device_pointer *p)
    {
        p->m=NULL;
        p->offset=0;
        p->_null=true;
    }

    static void null(void* p)
    {
        device_pointer* d=(device_pointer*)p;
        d->m=NULL;
        d->offset=0;
        d->_null=true;
    }

    static bool isNull(device_pointer p)
    {
        return p._null;
    }

};

}

}

}

#endif
