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


sofa::helper::OpenCLProgram* OpenCLMemoryManager_program;
void OpenCLMemoryManager_CreateProgram()
{
    if(OpenCLMemoryManager_program==NULL)
    {

        std::string source =*sofa::helper::OpenCLProgram::loadSource("OpenCLMemoryManager.cl");
        source = stringBSIZE + source;

        OpenCLMemoryManager_program
            = new sofa::helper::OpenCLProgram(&source);
        OpenCLMemoryManager_program->buildProgram();
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
        std::cout << OpenCLMemoryManager_program->buildLog(0);
    }
}

sofa::helper::OpenCLKernel * OpenCLMemoryManager_memsetDevice_kernel;


void OpenCLMemoryManager_memsetDevice(int d, _device_pointer a, int value, size_t size)
{

    DEBUG_TEXT("OpenCLMemoryManager_memsetDevice");
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;

    unsigned int i;
    unsigned int offset;

    OpenCLMemoryManager_CreateProgram();

    if(OpenCLMemoryManager_memsetDevice_kernel==NULL)OpenCLMemoryManager_memsetDevice_kernel
            = new sofa::helper::OpenCLKernel(OpenCLMemoryManager_program,"MemoryManager_memset");

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
