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
#include "OpenCLTypes.h"
#include "myopencl.h"
#include "OpenCLMemoryManager.h"
#include "OpenCLProgram.h"
#include "OpenCLKernel.h"
#include "OpenCLMemoryManager.h"

#include "OpenCLMechanicalObject.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/container/MappedObject.inl>

//#include "tools/top.h"


#define DEBUG_TEXT(t) //printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);

namespace sofa
{

namespace component
{

namespace container
{
// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.

template class MechanicalObject<gpu::opencl::OpenCLVec3fTypes>;
template class MechanicalObject<gpu::opencl::OpenCLVec3f1Types>;
template class MechanicalObject<gpu::opencl::OpenCLRigid3fTypes>;

template class MechanicalObject<gpu::opencl::OpenCLVec3dTypes>;
template class MechanicalObject<gpu::opencl::OpenCLVec3d1Types>;
template class MechanicalObject<gpu::opencl::OpenCLRigid3dTypes>;

}

} // namespace component

namespace gpu
{

namespace opencl
{

SOFA_DECL_CLASS(OpenCLMechanicalObject)

int MechanicalObjectOpenCLClass = core::RegisterObject("Supports GPU-side computations using OpenCL")
        .add< component::container::MechanicalObject<OpenCLVec3fTypes> >()
        .add< component::container::MechanicalObject<OpenCLVec3f1Types> >()
        .add< component::container::MechanicalObject<OpenCLRigid3fTypes> >()
        .add< component::container::MechanicalObject<OpenCLVec3dTypes> >()
        .add< component::container::MechanicalObject<OpenCLVec3d1Types> >()
        .add< component::container::MechanicalObject<OpenCLRigid3dTypes> >()
        ;

int MappedObjectOpenCLClass = core::RegisterObject("Supports GPU-side computations using OpenCL")
        .add< component::container::MappedObject<OpenCLVec3fTypes> >()
        .add< component::container::MappedObject<OpenCLVec3f1Types> >()
        .add< component::container::MappedObject<OpenCLRigid3fTypes> >()
        .add< component::container::MappedObject<OpenCLVec3dTypes> >()
        .add< component::container::MappedObject<OpenCLVec3d1Types> >()
        .add< component::container::MappedObject<OpenCLRigid3dTypes> >()
        ;

////////////////////////////////////////////////////////////////////////////////////
//start kernel

sofa::helper::OpenCLProgram* MechanicalObjectOpenCLFloat_program;
sofa::helper::OpenCLProgram* MechanicalObjectOpenCLDouble_program;

void MechanicalObject_CreateProgramWithFloat()
{
    if(MechanicalObjectOpenCLFloat_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="float";
        types["Real4"]="float4";

        std::cout << sofa::helper::OpenCLProgram::loadSource("OpenCLMechanicalObject.cl") << std::endl;
        MechanicalObjectOpenCLFloat_program
            = new sofa::helper::OpenCLProgram(sofa::helper::OpenCLProgram::loadSource("OpenCLMechanicalObject.cl"),&types);

        MechanicalObjectOpenCLFloat_program->buildProgram();
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
        std::cout << MechanicalObjectOpenCLFloat_program->buildLog(0);
    }
}

void MechanicalObject_CreateProgramWithDouble()
{

    if(MechanicalObjectOpenCLDouble_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="double";
        types["Real4"]="double4";

        MechanicalObjectOpenCLDouble_program
            = new sofa::helper::OpenCLProgram(sofa::helper::OpenCLProgram::loadSource("OpenCLMechanicalObject.cl"),&types);

        MechanicalObjectOpenCLDouble_program->buildProgram();


    }
}

// vOp (2/4)


sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vOp_kernel;
void MechanicalObjectOpenCLVec3f_vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, float f)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    size*=3;
    DEBUG_TEXT( "MechanicalObjectOpenCLVec3f_vOp\t");

    MechanicalObject_CreateProgramWithFloat();
    if(MechanicalObjectOpenCLVec3f_vOp_kernel==NULL)MechanicalObjectOpenCLVec3f_vOp_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec1t_vOp");
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(0,&res);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(1,&a);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(2,&b);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<float>(3,&f);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3f_vOp_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

}



sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3d_vOp_kernel;
void MechanicalObjectOpenCLVec3d_vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, double f)
{
    size*=3;
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    DEBUG_TEXT( "MechanicalObjectOpenCLVec3d_vOp\t");
    MechanicalObject_CreateProgramWithDouble();
    if(MechanicalObjectOpenCLVec3d_vOp_kernel==NULL)MechanicalObjectOpenCLVec3d_vOp_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLDouble_program,"MechanicalObject_Vec1t_vOp");
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(0,&res);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(1,&a);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(2,&b);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<double>(3,&f);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3d_vOp_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
}

// vOp2

sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vOp_v2_kernel;
void MechanicalObjectOpenCLVec3f_vOp_v2(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, float f)
{

    size*=3;
    DEBUG_TEXT( "MechanicalObjectOpenCLVec3f_vOp_v2\t");



    MechanicalObject_CreateProgramWithFloat();
    if(MechanicalObjectOpenCLVec3f_vOp_v2_kernel==NULL)MechanicalObjectOpenCLVec3f_vOp_v2_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"Vec1t_vOp_v2");
    MechanicalObjectOpenCLVec3f_vOp_v2_kernel->setArg<_device_pointer>(0,&res);
    MechanicalObjectOpenCLVec3f_vOp_v2_kernel->setArg<_device_pointer>(1,&a);
    MechanicalObjectOpenCLVec3f_vOp_v2_kernel->setArg<_device_pointer>(2,&b);
    MechanicalObjectOpenCLVec3f_vOp_v2_kernel->setArg<float>(3,&f);

    size_t work_size[1];
    work_size[0]=size/4+1;

    MechanicalObjectOpenCLVec3f_vOp_v2_kernel->execute(0,1,NULL,work_size,NULL);	//note: num_device = const = 0

}

// vMEq (2/4)
/// @note no offset management

sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vMEq_kernel;
void MechanicalObjectOpenCLVec3f_vMEq(size_t size, _device_pointer res, float f)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    size*=3;
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vMEq");


    MechanicalObject_CreateProgramWithFloat();
    if(MechanicalObjectOpenCLVec3f_vMEq_kernel==NULL)MechanicalObjectOpenCLVec3f_vMEq_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec1t_vMEq");

    MechanicalObjectOpenCLVec3f_vMEq_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3f_vMEq_kernel->setArg<float>(1,&f);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3f_vMEq_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

}

sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3d_vMEq_kernel;
void MechanicalObjectOpenCLVec3d_vMEq(size_t size, _device_pointer res, double f)
{
    size*=3;
    DEBUG_TEXT("MechanicalObjectOpenCLVec3d_vMEq");



    MechanicalObject_CreateProgramWithDouble();
    if(MechanicalObjectOpenCLVec3d_vMEq_kernel==NULL)MechanicalObjectOpenCLVec3d_vMEq_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLDouble_program,"Vec1t_vMEq");
    MechanicalObjectOpenCLVec3d_vMEq_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3d_vMEq_kernel->setArg<double>(1,&f);

    size_t work_size[1];
    work_size[0]=size;

    MechanicalObjectOpenCLVec3d_vMEq_kernel->execute(0,1,NULL,work_size,NULL);	//note: num_device = const = 0

}




//vMeq2

sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vMEq_v2_kernel;
void MechanicalObjectOpenCLVec3f_vMEq_v2(size_t size, _device_pointer res, float f)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vMEq_v2");

    size*=3;

    MechanicalObject_CreateProgramWithFloat();

    if(MechanicalObjectOpenCLVec3f_vMEq_v2_kernel==NULL)MechanicalObjectOpenCLVec3f_vMEq_v2_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"Vec1t_vMEq_v2");

    MechanicalObjectOpenCLVec3f_vMEq_v2_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3f_vMEq_v2_kernel->setArg<float>(1,&f);

    size_t work_size[1];
    work_size[0]=(size>>2) + 1;


    MechanicalObjectOpenCLVec3f_vMEq_v2_kernel->execute(0,1,NULL,work_size,NULL);	//note: num_device = const = 0
}












// vClear (4/4)

sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vClear_kernel;
void MechanicalObjectOpenCLVec3f_vClear(size_t size, _device_pointer res)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    size*=3;
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vClear");

    MechanicalObject_CreateProgramWithFloat();

    if(MechanicalObjectOpenCLVec3f_vClear_kernel==NULL)MechanicalObjectOpenCLVec3f_vClear_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec1t_vClear");

    MechanicalObjectOpenCLVec3f_vClear_kernel->setArg<cl_mem>(0,&(res.m));


    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3f_vClear_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0*/


}

void MechanicalObjectOpenCLVec3d_vClear(size_t size, _device_pointer res)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3d_vClear");
    OpenCLMemoryManager<double>::memsetDevice(0,res, 0, size*3*sizeof(double));
}

void MechanicalObjectOpenCLVec3f1_vClear(size_t size, _device_pointer res)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f1_vClear");
    OpenCLMemoryManager<float>::memsetDevice(0,res, 0, size*4*sizeof(float));
}

void MechanicalObjectOpenCLVec3d1_vClear(size_t size, _device_pointer res)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3d1_vClear");
    OpenCLMemoryManager<double>::memsetDevice(0,res, 0, size*4*sizeof(double));
}

// vAssign (4/4)

void MechanicalObjectOpenCLVec3f_vAssign(size_t size, _device_pointer res, const _device_pointer a)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vAssign");
    OpenCLMemoryManager<float>::memcpyDeviceToDevice(0,res,a,size*3*sizeof(float));
}

void MechanicalObjectOpenCLVec3f1_vAssign(size_t size, _device_pointer res, const _device_pointer a)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f1_vAssign");
    OpenCLMemoryManager<float>::memcpyDeviceToDevice(0,res,a,size*4*sizeof(float));
}

void MechanicalObjectOpenCLVec3d_vAssign(size_t size, _device_pointer res, const _device_pointer a)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3d_vAssign");
    OpenCLMemoryManager<double>::memcpyDeviceToDevice(0,res,a,size*3*sizeof(double));
}

void MechanicalObjectOpenCLVec3d1_vAssign(size_t size, _device_pointer res, const _device_pointer a)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3d1_vAssign");
    OpenCLMemoryManager<double>::memcpyDeviceToDevice(0,res,a,size*4*sizeof(double));
}

// vEqBF (1/4)
/// @note no tested

sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vEqBF_kernel;
void MechanicalObjectOpenCLVec3f_vEqBF(size_t size, _device_pointer res, const _device_pointer b, float f)
{
    size*=3;
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vEqBF");
    MechanicalObject_CreateProgramWithFloat();


    if(MechanicalObjectOpenCLVec3f_vEqBF_kernel==NULL)MechanicalObjectOpenCLVec3f_vEqBF_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"Vec1t_vEqBF");

    MechanicalObjectOpenCLVec3f_vEqBF_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3f_vEqBF_kernel->setArg<cl_mem>(1,&(b.m));
    MechanicalObjectOpenCLVec3f_vEqBF_kernel->setArg<float>(2,&f);

    size_t work_size[1];
    work_size[0]=size;

    MechanicalObjectOpenCLVec3f_vEqBF_kernel->execute(0,1,NULL,work_size,NULL);	//note: num_device = const = 0

}

// vPEqBF (1/4)

sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vPEqBF_kernel;
void MechanicalObjectOpenCLVec3f_vPEqBF(size_t size, _device_pointer res, const _device_pointer b, float f)
{
    size*=3;
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vPEqBF");
    MechanicalObject_CreateProgramWithFloat();

    if(MechanicalObjectOpenCLVec3f_vPEqBF_kernel==NULL)MechanicalObjectOpenCLVec3f_vPEqBF_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"Vec1t_vPEqBF");

    MechanicalObjectOpenCLVec3f_vPEqBF_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3f_vPEqBF_kernel->setArg<cl_mem>(1,&(b.m));
    MechanicalObjectOpenCLVec3f_vPEqBF_kernel->setArg<float>(2,&f);

    size_t work_size[1];
    work_size[0]=size;

    MechanicalObjectOpenCLVec3f_vPEqBF_kernel->execute(0,1,NULL,work_size,NULL);	//note: num_device = const = 0

}


// vDot (1/4)
#define RED_SIZE 512

sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vDot_kernel;
void MechanicalObjectOpenCLVec3f_vDot(size_t size, float* res, const _device_pointer a, const _device_pointer b, _device_pointer tmp, float* cputmp)
{
    size*=3;
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vDot");

    MechanicalObject_CreateProgramWithFloat();

    if(MechanicalObjectOpenCLVec3f_vDot_kernel==NULL)MechanicalObjectOpenCLVec3f_vDot_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"Vec1t_vDot");

    int s=size;
    MechanicalObjectOpenCLVec3f_vDot_kernel->setArg<int>(0,&s);
    MechanicalObjectOpenCLVec3f_vDot_kernel->setArg<cl_mem>(1,&(tmp.m));
    MechanicalObjectOpenCLVec3f_vDot_kernel->setArg<cl_mem>(2,&(a.m));
    MechanicalObjectOpenCLVec3f_vDot_kernel->setArg<cl_mem>(3,&(b.m));

    size_t work_size[1],local_size[1];
    local_size[0]= RED_SIZE;
    work_size[0]= (size%4==0)?size/4:size/4 +1;
    if((work_size[0]%RED_SIZE)!=0)work_size[0]=(work_size[0]/RED_SIZE +1)*RED_SIZE;

    MechanicalObjectOpenCLVec3f_vDot_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
    OpenCLMemoryManager<float>::memcpyDeviceToHost(0,cputmp,tmp, (work_size[0]/RED_SIZE)*sizeof(float));

    *res=0;
    for(unsigned int i=0; i< work_size[0]/RED_SIZE; i++)*res+=cputmp[i];

}

// vDotTmpSize (1/4)

int MechanicalObjectOpenCLVec3f_vDotTmpSize(size_t size)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vDotTmpSize");

    size *= 3;
    int nblocs = (size+RED_SIZE-1)/RED_SIZE;
    if (nblocs > 256) nblocs = 256;
    return nblocs;
}

// vAdd Vec1t_vAdd

sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vAdd_kernel;
void MechanicalObjectOpenCLVec3f_vAdd(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b)
{
    size*=3;
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vAdd");



    MechanicalObject_CreateProgramWithFloat();
    if(MechanicalObjectOpenCLVec3f_vAdd_kernel==NULL)MechanicalObjectOpenCLVec3f_vAdd_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"Vec1t_vAdd");
    MechanicalObjectOpenCLVec3f_vAdd_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3f_vAdd_kernel->setArg<cl_mem>(1,&(a.m));
    MechanicalObjectOpenCLVec3f_vAdd_kernel->setArg<cl_mem>(2,&(b.m));


    size_t work_size[1];
    work_size[0]=size;

    MechanicalObjectOpenCLVec3f_vAdd_kernel->execute(0,1,NULL,work_size,NULL);	//note: num_device = const = 0

}

sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vPEq_kernel;
void MechanicalObjectOpenCLVec3f_vPEq(size_t size, _device_pointer res, const _device_pointer a)
{
    size*=3;
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vPEq") ;


    MechanicalObject_CreateProgramWithFloat();
    if(MechanicalObjectOpenCLVec3f_vPEq_kernel==NULL)MechanicalObjectOpenCLVec3f_vPEq_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"Vec1t_vPEq");
    MechanicalObjectOpenCLVec3f_vPEq_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3f_vPEq_kernel->setArg<cl_mem>(1,&(a.m));

    size_t work_size[1];
    work_size[0]=size;

    MechanicalObjectOpenCLVec3f_vPEq_kernel->execute(0,1,NULL,work_size,NULL);	//note: num_device = const = 0*/

}


sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vPEqBF2_kernel;
void MechanicalObjectOpenCLVec3f_vPEqBF2(size_t size, _device_pointer res1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer b2, float f2)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vPEqBF2");

    size*=3;

    MechanicalObject_CreateProgramWithFloat();


    if(MechanicalObjectOpenCLVec3f_vPEqBF2_kernel==NULL)MechanicalObjectOpenCLVec3f_vPEqBF2_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec1t_vPEqBF2");

    MechanicalObjectOpenCLVec3f_vPEqBF2_kernel->setArg<cl_mem>(0,&(res1.m));
    MechanicalObjectOpenCLVec3f_vPEqBF2_kernel->setArg<cl_mem>(1,&(b1.m));
    MechanicalObjectOpenCLVec3f_vPEqBF2_kernel->setArg<float>(2,&f1);
    MechanicalObjectOpenCLVec3f_vPEqBF2_kernel->setArg<cl_mem>(3,&(res2.m));
    MechanicalObjectOpenCLVec3f_vPEqBF2_kernel->setArg<cl_mem>(4,&(b2.m));
    MechanicalObjectOpenCLVec3f_vPEqBF2_kernel->setArg<float>(5,&f2);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3f_vPEqBF2_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0


}

sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vPEqBF2_v2_kernel;
void MechanicalObjectOpenCLVec3f_vPEqBF2_v2(size_t size, _device_pointer res1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer b2, float f2)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vPEqBF2_v2");

    size*=3;

    MechanicalObject_CreateProgramWithFloat();


    if(MechanicalObjectOpenCLVec3f_vPEqBF2_v2_kernel==NULL)MechanicalObjectOpenCLVec3f_vPEqBF2_v2_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"Vec1t_vPEqBF2_v2");

    MechanicalObjectOpenCLVec3f_vPEqBF2_v2_kernel->setArg<cl_mem>(0,&(res1.m));
    MechanicalObjectOpenCLVec3f_vPEqBF2_v2_kernel->setArg<cl_mem>(1,&(b1.m));
    MechanicalObjectOpenCLVec3f_vPEqBF2_v2_kernel->setArg<float>(2,&f1);
    MechanicalObjectOpenCLVec3f_vPEqBF2_v2_kernel->setArg<cl_mem>(3,&(res2.m));
    MechanicalObjectOpenCLVec3f_vPEqBF2_v2_kernel->setArg<cl_mem>(4,&(b2.m));
    MechanicalObjectOpenCLVec3f_vPEqBF2_v2_kernel->setArg<float>(5,&f2);

    size_t work_size[1];
    work_size[0]=size/4;

    MechanicalObjectOpenCLVec3f_vPEqBF2_v2_kernel->execute(0,1,NULL,work_size,NULL);	//note: num_device = const = 0


}


sofa::helper::OpenCLKernel * MechanicalObjectOpenCLVec3f_vIntegrate_kernel;
void MechanicalObjectOpenCLVec3f_vIntegrate(size_t size, const _device_pointer a, _device_pointer v, _device_pointer x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vIntegrate");

    size*=3;
    MechanicalObject_CreateProgramWithFloat();


    if(MechanicalObjectOpenCLVec3f_vIntegrate_kernel==NULL)MechanicalObjectOpenCLVec3f_vIntegrate_kernel
            = new sofa::helper::OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec1t_vIntegrate");

    MechanicalObjectOpenCLVec3f_vIntegrate_kernel->setArg<cl_mem>(0,&(a.m));
    MechanicalObjectOpenCLVec3f_vIntegrate_kernel->setArg<cl_mem>(1,&(v.m));
    MechanicalObjectOpenCLVec3f_vIntegrate_kernel->setArg<cl_mem>(2,&(x.m));
    MechanicalObjectOpenCLVec3f_vIntegrate_kernel->setArg<float>(3,&f_v_v);
    MechanicalObjectOpenCLVec3f_vIntegrate_kernel->setArg<float>(4,&f_v_a);
    MechanicalObjectOpenCLVec3f_vIntegrate_kernel->setArg<float>(5,&f_x_x);
    MechanicalObjectOpenCLVec3f_vIntegrate_kernel->setArg<float>(6,&f_x_v);


    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);


    MechanicalObjectOpenCLVec3f_vIntegrate_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

}








void MechanicalObjectOpenCLVec3f_vPEq4BF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b11*/, float /*f11*/, const _device_pointer /*b12*/, float /*f12*/, const _device_pointer /*b13*/, float /*f13*/, const _device_pointer /*b14*/, float /*f14*/,
        _device_pointer /*res2*/, const _device_pointer /*b21*/, float /*f21*/, const _device_pointer /*b22*/, float /*f22*/, const _device_pointer /*b23*/, float /*f23*/, const _device_pointer /*b24*/, float /*f24*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3f_vOp2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*a1*/, const _device_pointer /*b1*/, float /*f1*/, _device_pointer /*res2*/, const _device_pointer /*a2*/, const _device_pointer /*b2*/, float /*f2*/) {DEBUG_TEXT("no implemented");}


int MultiMechanicalObjectOpenCLVec3f_vDotTmpSize(size_t  /*n*/, VDotOp*  /*ops*/) {DEBUG_TEXT("no implemented"); return 0;}
void MultiMechanicalObjectOpenCLVec3f_vDot(size_t  /*n*/, VDotOp*  /*ops*/, double*  /*results*/, _device_pointer /*tmp*/, float*  /*cputmp*/) {DEBUG_TEXT("no implemented");}


void MultiMechanicalObjectOpenCLVec3f_vOp(size_t  /*n*/, VOpF*  /*ops*/) {DEBUG_TEXT("no implemented");}


void MultiMechanicalObjectOpenCLVec3f_vClear(size_t  /*n*/, VClearOp*  /*ops*/) {DEBUG_TEXT("no implemented");}



void MechanicalObjectOpenCLVec3f1_vMEq(size_t /*size*/, _device_pointer /*res*/, float /*f*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3f1_vEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, float /*f*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3f1_vPEq(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3f1_vPEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, float /*f*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3f1_vAdd(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3f1_vOp(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/, float /*f*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3f1_vIntegrate(size_t /*size*/, const _device_pointer /*a*/, _device_pointer /*v*/, _device_pointer /*x*/, float /*f_v_v*/, float /*f_v_a*/, float /*f_x_x*/, float /*f_x_v*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3f1_vPEqBF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b1*/, float /*f1*/, _device_pointer /*res2*/, const _device_pointer /*b2*/, float /*f2*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3f1_vPEq4BF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b11*/, float /*f11*/, const _device_pointer /*b12*/, float /*f12*/, const _device_pointer /*b13*/, float /*f13*/, const _device_pointer /*b14*/, float /*f14*/,
        _device_pointer /*res2*/, const _device_pointer /*b21*/, float /*f21*/, const _device_pointer /*b22*/, float /*f22*/, const _device_pointer /*b23*/, float /*f23*/, const _device_pointer /*b24*/, float /*f24*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3f1_vOp2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*a1*/, const _device_pointer /*b1*/, float /*f1*/, _device_pointer /*res2*/, const _device_pointer /*a2*/, const _device_pointer /*b2*/, float /*f2*/) {DEBUG_TEXT("no implemented");}
int MechanicalObjectOpenCLVec3f1_vDotTmpSize(size_t /*size*/) {DEBUG_TEXT("no implemented"); return 0;}
void MechanicalObjectOpenCLVec3f1_vDot(size_t /*size*/, float* /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/, _device_pointer /*tmp*/, float*  /*cputmp*/) {DEBUG_TEXT("no implemented");}





void MechanicalObjectOpenCLVec3d_vEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, double /*f*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d_vPEq(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d_vPEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, double /*f*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d_vAdd(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d_vIntegrate(size_t /*size*/, const _device_pointer /*a*/, _device_pointer /*v*/, _device_pointer /*x*/, double /*f_v_v*/, double /*f_v_a*/, double /*f_x_x*/, double /*f_x_v*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d_vPEqBF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b1*/, double /*f1*/, _device_pointer /*res2*/, const _device_pointer /*b2*/, double /*f2*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d_vPEq4BF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b11*/, double /*f11*/, const _device_pointer /*b12*/, double /*f12*/, const _device_pointer /*b13*/, double /*f13*/, const _device_pointer /*b14*/, double /*f14*/,
        _device_pointer /*res2*/, const _device_pointer /*b21*/, double /*f21*/, const _device_pointer /*b22*/, double /*f22*/, const _device_pointer /*b23*/, double /*f23*/, const _device_pointer /*b24*/, double /*f24*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d_vOp2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*a1*/, const _device_pointer /*b1*/, double /*f1*/, _device_pointer /*res2*/, const _device_pointer /*a2*/, const _device_pointer /*b2*/, double /*f2*/) {DEBUG_TEXT("no implemented");}
int MechanicalObjectOpenCLVec3d_vDotTmpSize(size_t /*size*/) {DEBUG_TEXT("no implemented"); return 0;}
void MechanicalObjectOpenCLVec3d_vDot(size_t /*size*/, double* /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/, _device_pointer /*tmp*/, double*  /*cputmp*/) {DEBUG_TEXT("no implemented");}

void MechanicalObjectOpenCLVec3d1_vMEq(size_t /*size*/, _device_pointer /*res*/, double /*f*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d1_vEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, double /*f*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d1_vPEq(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d1_vPEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, double /*f*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d1_vAdd(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d1_vOp(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/, double /*f*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d1_vIntegrate(size_t /*size*/, const _device_pointer /*a*/, _device_pointer /*v*/, _device_pointer /*x*/, double /*f_v_v*/, double /*f_v_a*/, double /*f_x_x*/, double /*f_x_v*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d1_vPEqBF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b1*/, double /*f1*/, _device_pointer /*res2*/, const _device_pointer /*b2*/, double /*f2*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d1_vPEq4BF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b11*/, double /*f11*/, const _device_pointer /*b12*/, double /*f12*/, const _device_pointer /*b13*/, double /*f13*/, const _device_pointer /*b14*/, double /*f14*/,
        _device_pointer /*res2*/, const _device_pointer /*b21*/, double /*f21*/, const _device_pointer /*b22*/, double /*f22*/, const _device_pointer /*b23*/, double /*f23*/, const _device_pointer /*b24*/, double /*f24*/) {DEBUG_TEXT("no implemented");}
void MechanicalObjectOpenCLVec3d1_vOp2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*a1*/, const _device_pointer /*b1*/, double /*f1*/, _device_pointer /*res2*/, const _device_pointer /*a2*/, const _device_pointer /*b2*/, double /*f2*/) {DEBUG_TEXT("no implemented");}
int MechanicalObjectOpenCLVec3d1_vDotTmpSize(size_t /*size*/) {DEBUG_TEXT("no implemented"); return 0;}
void MechanicalObjectOpenCLVec3d1_vDot(size_t /*size*/, double* /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/, _device_pointer /*tmp*/, double*  /*cputmp*/) {DEBUG_TEXT("no implemented");}



























} // namespace OpenCL

} // namespace gpu

} // namespace sofa


#undef DEBUG_TEXT
