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
#ifndef SOFAOPENCL_MYOPENCL_H
#define SOFAOPENCL_MYOPENCL_H


#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "config.h"
#include <string>


#define ERROR_OFFSET(t) {if(t.offset!=0){printf("Error Offset %s %d: %s %d\n",__FILE__,__LINE__,#t,(int)t.offset);exit(-1);}}
#define NOT_IMPLEMENTED() //{printf("Not implemented %s %d\n",__FILE__,__LINE__);exit(-1);}
#define BARRIER(x,y,z) //myopenclBarrier(x,y,z);


#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace opencl
{
#endif



extern "C" {


    typedef struct _device_pointer
    {
        mutable cl_mem m;
        size_t offset;
        bool _null;
        _device_pointer()
        {
            m = NULL;
            offset = 0;
            _null=true;
        }
    } _device_pointer;


    SOFAOPENCL_API extern int myopenclInit(int device=-1);
    SOFAOPENCL_API extern int myopenclGetnumDevices();
    SOFAOPENCL_API extern void myopenclCreateBuffer(int device,cl_mem* dptr,int n);
    SOFAOPENCL_API extern void myopenclReleaseBuffer(int device,cl_mem p);
    SOFAOPENCL_API extern void myopenclEnqueueWriteBuffer(int device,cl_mem ddest,size_t offset,const void * hsrc,size_t n);
    SOFAOPENCL_API extern void myopenclEnqueueReadBuffer(int device,void * hdest,const cl_mem dsrc,size_t offset, size_t n);
    SOFAOPENCL_API extern void myopenclEnqueueCopyBuffer(int device, cl_mem ddest,size_t destOffset, const cl_mem dsrc,size_t srcOffset, size_t n);
    SOFAOPENCL_API extern void myopenclSetKernelArg(cl_kernel kernel, int num_arg,int size,void* arg);

    SOFAOPENCL_API extern bool myopenclBuildProgram(void* p);
    SOFAOPENCL_API extern bool myopenclBuildProgramWithFlags(void * program, char * flags);
    SOFAOPENCL_API extern cl_program myopenclProgramWithSource(const char * s,const size_t size);
    SOFAOPENCL_API extern cl_kernel myopenclCreateKernel(void* p,const char * kernel_name);
    SOFAOPENCL_API extern void myopenclExecKernel(int device,cl_kernel kernel,unsigned int work_dim,const size_t *global_work_offset,const size_t *global_work_size,const size_t *local_work_size);
    SOFAOPENCL_API extern void myopenclBarrier(_device_pointer d,std::string file, int line);

    SOFAOPENCL_API extern void myopenclMemsetDevice(int d, _device_pointer dDestPointer, int value, size_t n);
    SOFAOPENCL_API extern void* myopencldevice(int device);
    SOFAOPENCL_API extern int myopenclNumDevices();
    SOFAOPENCL_API extern int myopenclError();
    SOFAOPENCL_API extern const char* myopenclErrorMsg(cl_int err);
    SOFAOPENCL_API extern void myopenclShowError(std::string file, int line);
    SOFAOPENCL_API extern const char* myopenclPath();


    enum MyopenclVerboseLevel
    {
        LOG_NONE = 0,
        LOG_ERR = 1,
        LOG_INFO = 2,
        LOG_TRACE = 3
    };

    extern MyopenclVerboseLevel SOFAOPENCL_API myopenclVerboseLevel;
}

template<class T>
void myopenclSetKernelArg(cl_kernel kernel, int num_arg, const T* arg);

template<class T>
extern inline void myopenclSetKernelArg(cl_kernel kernel, int num_arg, const T* arg)
{
    myopenclSetKernelArg(kernel, num_arg, sizeof(T), (void*)arg);
}

template<>
void myopenclSetKernelArg<_device_pointer>(cl_kernel kernel, int num_arg, const _device_pointer* arg);


#if defined(__cplusplus)
} // namespace opencl
} // namespace gpu
} // namespace sofa
#endif

#endif



