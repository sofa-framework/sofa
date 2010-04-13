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
void CreateProgram()
{
    if(OpenCLMemoryManager_program==NULL)
    {
        OpenCLMemoryManager_program
            = new sofa::helper::OpenCLProgram(sofa::helper::OpenCLProgram::loadSource("OpenCLMemoryManager.cl"));
        OpenCLMemoryManager_program->buildProgram();
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
        std::cout << OpenCLMemoryManager_program->buildLog(0);
    }
}

sofa::helper::OpenCLKernel * OpenCLMemoryManager_memsetDevice_kernel;


void OpenCLMemoryManager_memsetDevice(int d, _device_pointer a, int value, size_t size)
{
    unsigned int i;
    unsigned int offset;
    DEBUG_TEXT("OpenCLMemoryManager_memsetDevice");
//std::cout << a.m << " " << "\n" << "offset"<< a.offset << " "<< value << " " << size << "\n";

//	top(0);
    CreateProgram();
//	top(1);
    if(OpenCLMemoryManager_memsetDevice_kernel==NULL)OpenCLMemoryManager_memsetDevice_kernel
            = new sofa::helper::OpenCLKernel(OpenCLMemoryManager_program,"memset");
//	top(2);


    i= value;
    i= (i << 8) + value;
    i= (i << 8) + value;
    i= (i << 8) + value;

    offset = a.offset/(4*sizeof(int));


    OpenCLMemoryManager_memsetDevice_kernel->setArg<cl_mem>(0,&(a.m));
//	top(3);
    OpenCLMemoryManager_memsetDevice_kernel->setArg<unsigned int>(1,(unsigned int*)&(offset));
//	top(4);
    OpenCLMemoryManager_memsetDevice_kernel->setArg<unsigned int>(2,(unsigned int*)&i);
//	top(5);
    size_t work_size[1];
    work_size[0]=size/(4*sizeof(int));
//	top(6);
    OpenCLMemoryManager_memsetDevice_kernel->execute(d,1,NULL,work_size,NULL);	//note: num_device = const = 0*/

    /*myopenclMemsetDevice(d,a,value,size);

    	top(7);
    	float ra[10];
    	myopenclEnqueueReadBuffer(d,ra,a.m,0,10);
    	for(int j=0;j<10;j++)printf("%f\n",ra[j]);
    	top(8);*/

//	topLog();

}



}
}
}
