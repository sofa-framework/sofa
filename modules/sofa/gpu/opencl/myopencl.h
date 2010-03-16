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
#include <string.h>

#if defined(__cplusplus)
namespace sofa
{
namespace gpu
{
namespace opencl
{
#endif

extern "C" {
    /*	extern int _numDevices;
    	extern cl_context _context;
    	extern cl_command_queue* _queues;
    	extern cl_device_id* _devices;
    	extern cl_int _error;*/


    extern int myopenclInit(int device=-1);
    extern int myopenclGetnumDevices();

// 	extern bool myaddQueue(cl_command_queue queue);
// 	extern cl_command_queue myqueue(int i);
// 	extern cl_device_id mydevice(int i);
// 	extern cl_context mycontext();
// 	extern void releaseContext();
// 	extern void releaseQueues();
// 	extern void releaseDevices();
// 	extern cl_context createContext(cl_device_type type);
// 	extern void createDevices();
// 	extern void createQueues();
// 	extern int numDevices();
// 	extern cl_int & error();
// 	extern void showError(std::string file, int line);
}


#if defined(__cplusplus)
} // namespace cuda
} // namespace gpu
} // namespace sofa
#endif

#endif



