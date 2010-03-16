#ifndef OPENCLKERNEL_H
#define OPENCLKERNEL_H

#include <string>
#include "OpenCLProgramParser.h"
#include "OpenCLManager.h"
#include "OpenCLProgram.h"
#include <CL/cl.h>

namespace sofa
{

namespace helper
{

class OpenCLKernel
{
    cl_kernel _kernel;

public:
    OpenCLKernel(OpenCLProgram &p, const char *kernel_name)
    {
        _kernel= clCreateKernel(p.program(), kernel_name, &OpenCLManager::error());
    }

    cl_kernel kernel() {return _kernel;}

    void setArg(int num_arg, cl_mem mem)
    {
        OpenCLManager::error() = clSetKernelArg(_kernel, num_arg,sizeof(cl_mem), (void *)(&mem));
    }
};

}

}


#endif // OPENCLKERNEL_H
