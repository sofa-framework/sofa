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
#ifndef MYOPENCL_H
#define MYOPENCL_H

#include "gpuopencl.h"
#include <string>
#include <CL/cl.h>

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace opencl
{
#endif

extern "C" {

    extern int myopenclInit(int device=-1);
    extern int myopenclGetnumDevices();
    extern cl_mem myopenclCreateBuffer(int n);
    extern void myopenclReleaseBuffer(cl_mem p);
    extern void myopenclEnqueueWriteBuffer(int device,cl_mem dest,void* src,size_t n);
    extern void myopenclEnqueueReadBuffer(int device,void* dest,cl_mem src, size_t n);
    extern void myopenclEnqueueCopyBuffer(int device, cl_mem dest, cl_mem src, size_t n);

    extern int myopenclNumDevices();
    extern cl_int & myopenclError();
    extern void myopenclShowError(std::string file, int line);
}


#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif

#endif



