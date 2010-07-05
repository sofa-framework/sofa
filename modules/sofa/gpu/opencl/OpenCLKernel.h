#ifndef OPENCLKERNEL_H
#define OPENCLKERNEL_H

#include <string>
#include "OpenCLProgramParser.h"
#include "myopencl.h"
#include "OpenCLProgram.h"

namespace sofa
{

namespace helper
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


#endif // OPENCLKERNEL_H
