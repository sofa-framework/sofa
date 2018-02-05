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
#ifndef SOFAOPENCL_OPENCLKERNEL_H
#define SOFAOPENCL_OPENCLKERNEL_H

#include <string>
#include "myopencl.h"
#include "OpenCLProgram.h"

namespace sofa
{

namespace gpu
{

namespace opencl
{

class OpenCLKernel
{
    cl_kernel _kernel;
    std::string _kernel_name;
public:
    OpenCLKernel(OpenCLProgram *p, const char *kernel_name)
    {
        _kernel = sofa::gpu::opencl::myopenclCreateKernel(p->program(), kernel_name);
        _kernel_name = kernel_name;
    }

    cl_kernel kernel() {return _kernel;}



    template <typename T>
    void setArg(int numArg,const T* arg)
    {
        //sofa::gpu::opencl::myopenclSetKernelArg(_kernel,numArg,sizeof(T),(void *)arg);
        sofa::gpu::opencl::myopenclSetKernelArg(_kernel,numArg,arg);
    }

    void setArg(int numArg,int size,void* arg)
    {
        sofa::gpu::opencl::myopenclSetKernelArg(_kernel,numArg,size,arg);
    }

//note: 'global_work_offset' must currently be a NULL value. In a future revision of OpenCL, global_work_offset can be used to specify an array of work_dim unsigned values that describe the offset used to calculate the global ID of a work-item instead of having the global IDs always start at offset (0, 0,... 0).
    void execute(int device, unsigned int work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size)
    {
        sofa::gpu::opencl::myopenclExecKernel(device,_kernel,work_dim,global_work_offset,global_work_size,local_work_size);
    }


};


}

}

}


#endif // SOFAOPENCL_OPENCLKERNEL_H
