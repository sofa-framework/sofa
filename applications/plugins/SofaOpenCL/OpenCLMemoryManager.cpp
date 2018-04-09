/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "OpenCLMemoryManager.h"
#include "OpenCLProgram.h"
#include "OpenCLKernel.h"
#include "myopencl.h"


//#include "tools/top.h"



namespace sofa
{

namespace gpu
{

namespace opencl
{

#define DEBUG_TEXT(t) //printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);


OpenCLProgram* OpenCLMemoryManager_program = NULL;
void OpenCLMemoryManager_CreateProgram()
{
    if(OpenCLMemoryManager_program==NULL)
    {
        OpenCLMemoryManager_program
            = new OpenCLProgram("OpenCLMemoryManager.cl",stringBSIZE);
        OpenCLMemoryManager_program->buildProgram();
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
        std::cout << OpenCLMemoryManager_program->buildLog(0);
    }
}

OpenCLKernel * OpenCLMemoryManager_memsetDevice_kernel = NULL;


void OpenCLMemoryManager_memsetDevice(int d, _device_pointer a, int value, size_t size)
{

    DEBUG_TEXT("OpenCLMemoryManager_memsetDevice");
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;

    unsigned int i;
    unsigned int offset;

    OpenCLMemoryManager_CreateProgram();

    if(OpenCLMemoryManager_memsetDevice_kernel==NULL)OpenCLMemoryManager_memsetDevice_kernel
            = new OpenCLKernel(OpenCLMemoryManager_program,"MemoryManager_memset");

    i= value;

    offset = a.offset/(sizeof(int));
    size = size/(sizeof(int));


    OpenCLMemoryManager_memsetDevice_kernel->setArg<cl_mem>(0,&(a.m));

    OpenCLMemoryManager_memsetDevice_kernel->setArg<unsigned int>(1,(unsigned int*)&(offset));

    OpenCLMemoryManager_memsetDevice_kernel->setArg<unsigned int>(2,(unsigned int*)&i);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    OpenCLMemoryManager_memsetDevice_kernel->execute(d,1,NULL,work_size,local_size);


    DEBUG_TEXT("~OpenCLMemoryManager_memsetDevice");
}



}
}
}
