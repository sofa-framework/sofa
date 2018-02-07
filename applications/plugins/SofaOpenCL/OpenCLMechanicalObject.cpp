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
#include "OpenCLTypes.h"
#include "OpenCLMemoryManager.h"
#include "OpenCLProgram.h"
#include "OpenCLKernel.h"
#include "OpenCLMemoryManager.h"

#include "OpenCLMechanicalObject.inl"
#include <sofa/core/ObjectFactory.h>
#include <SofaBaseMechanics/MappedObject.inl>
#include <sofa/core/State.inl>

//#include "tools/top.h"


#define DEBUG_TEXT(t) //printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);

namespace sofa
{

namespace core {

template class State<gpu::opencl::OpenCLVec3fTypes>;
template class State<gpu::opencl::OpenCLVec3f1Types>;
template class State<gpu::opencl::OpenCLRigid3fTypes>;

template class State<gpu::opencl::OpenCLVec3dTypes>;
template class State<gpu::opencl::OpenCLVec3d1Types>;
template class State<gpu::opencl::OpenCLRigid3dTypes>;

}

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

OpenCLProgram* MechanicalObjectOpenCLFloat_program = NULL;


OpenCLKernel * MechanicalObjectOpenCLVec3f_vOp_kernel = NULL;
OpenCLKernel * MechanicalObjectOpenCLVec3f_vMEq_kernel = NULL;
OpenCLKernel * MechanicalObjectOpenCLVec3f_vClear_kernel = NULL;
OpenCLKernel * MechanicalObjectOpenCLVec3f_vEqBF_kernel = NULL;
OpenCLKernel * MechanicalObjectOpenCLVec3f_vPEqBF_kernel = NULL;
OpenCLKernel * MechanicalObjectOpenCLVec3f_vDot_kernel = NULL;
OpenCLKernel * MechanicalObjectOpenCLVec3f_vAdd_kernel = NULL;
OpenCLKernel * MechanicalObjectOpenCLVec3f_vPEq_kernel = NULL;
OpenCLKernel * MechanicalObjectOpenCLVec3f_vPEqBF2_kernel = NULL;
OpenCLKernel * MechanicalObjectOpenCLVec3f_vIntegrate_kernel = NULL;
OpenCLKernel * MechanicalObjectOpenCLVec3f_vOp2_kernel = NULL;
OpenCLKernel * MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel = NULL;
void MechanicalObject_CreateProgramWithFloat()
{
    if(MechanicalObjectOpenCLFloat_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="float";
        types["Real4"]="float4";

        MechanicalObjectOpenCLFloat_program
            = new OpenCLProgram("OpenCLMechanicalObject.cl",stringBSIZE,&types);

        MechanicalObjectOpenCLFloat_program->buildProgram();
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
        std::cout << MechanicalObjectOpenCLFloat_program->buildLog(0);

        MechanicalObjectOpenCLVec3f_vOp_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec3t_vOp");

        MechanicalObjectOpenCLVec3f_vMEq_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec3t_vMEq");

        MechanicalObjectOpenCLVec3f_vClear_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec3t_vClear");

        MechanicalObjectOpenCLVec3f_vEqBF_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec3t_vEqBF");

        MechanicalObjectOpenCLVec3f_vPEqBF_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec3t_vPEqBF");

        MechanicalObjectOpenCLVec3f_vDot_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec1t_vDot");

        MechanicalObjectOpenCLVec3f_vAdd_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec3t_vAdd");

        MechanicalObjectOpenCLVec3f_vPEq_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec3t_vPEq");

        MechanicalObjectOpenCLVec3f_vPEqBF2_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec3t_vPEqBF2");

        MechanicalObjectOpenCLVec3f_vIntegrate_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec3t_vIntegrate");

        MechanicalObjectOpenCLVec3f_vOp2_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec3t_vOp2");

        MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLFloat_program,"MechanicalObject_Vec3t_vPEq4BF2");

    }
}

// vOp (2/4)


void MechanicalObjectOpenCLVec3f_vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, float f)
{
    DEBUG_TEXT( "MechanicalObjectOpenCLVec3f_vOp\t");
    BARRIER(a,__FILE__,__LINE__);

    ERROR_OFFSET(res)
    ERROR_OFFSET(a)
    ERROR_OFFSET(b)


    const int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;



    MechanicalObject_CreateProgramWithFloat();

    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(0,&res);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(1,&a);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(2,&b);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<float>(3,&f);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    BARRIER(a,__FILE__,__LINE__);
    MechanicalObjectOpenCLVec3f_vOp_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
}



// vMEq


void MechanicalObjectOpenCLVec3f_vMEq(size_t size, _device_pointer res, float f)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vMEq");
    BARRIER(res,__FILE__,__LINE__);
    ERROR_OFFSET(res)

    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
//size*=3;



    MechanicalObject_CreateProgramWithFloat();


    MechanicalObjectOpenCLVec3f_vMEq_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3f_vMEq_kernel->setArg<float>(1,&f);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    BARRIER(res,__FILE__,__LINE__);
    MechanicalObjectOpenCLVec3f_vMEq_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
}

// vClear (4/4)

void MechanicalObjectOpenCLVec3f_vClear(size_t size, _device_pointer res)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vClear");
    BARRIER(res,__FILE__,__LINE__);
    ERROR_OFFSET(res)

    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
//	size*=3;


    MechanicalObject_CreateProgramWithFloat();



    MechanicalObjectOpenCLVec3f_vClear_kernel->setArg(0,&res);


    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3f_vClear_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0*/
    BARRIER(res,__FILE__,__LINE__);
}

void MechanicalObjectOpenCLVec3d_vClear(size_t size, _device_pointer res)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3d_vClear");
    BARRIER(res,__FILE__,__LINE__);
    ERROR_OFFSET(res)

    OpenCLMemoryManager<double>::memsetDevice(0,res, 0, size*3*sizeof(double));
    BARRIER(res,__FILE__,__LINE__);
}

void MechanicalObjectOpenCLVec3f1_vClear(size_t size, _device_pointer res)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f1_vClear");
    ERROR_OFFSET(res)

    OpenCLMemoryManager<float>::memsetDevice(0,res, 0, size*4*sizeof(float));
}

void MechanicalObjectOpenCLVec3d1_vClear(size_t size, _device_pointer res)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3d1_vClear");
    ERROR_OFFSET(res)

    OpenCLMemoryManager<double>::memsetDevice(0,res, 0, size*4*sizeof(double));
}

// vAssign (4/4)

void MechanicalObjectOpenCLVec3f_vAssign(size_t size, _device_pointer res, const _device_pointer a)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vAssign");
    BARRIER(res,__FILE__,__LINE__);
    ERROR_OFFSET(res)

    OpenCLMemoryManager<float>::memcpyDeviceToDevice(0,res,a,size*3*sizeof(float));
    BARRIER(res,__FILE__,__LINE__);
}

void MechanicalObjectOpenCLVec3f1_vAssign(size_t size, _device_pointer res, const _device_pointer a)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f1_vAssign");
    ERROR_OFFSET(res)

    OpenCLMemoryManager<float>::memcpyDeviceToDevice(0,res,a,size*4*sizeof(float));
}

void MechanicalObjectOpenCLVec3d_vAssign(size_t size, _device_pointer res, const _device_pointer a)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3d_vAssign");
    ERROR_OFFSET(res)

    OpenCLMemoryManager<double>::memcpyDeviceToDevice(0,res,a,size*3*sizeof(double));
}

void MechanicalObjectOpenCLVec3d1_vAssign(size_t size, _device_pointer res, const _device_pointer a)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3d1_vAssign");
    ERROR_OFFSET(res)

    OpenCLMemoryManager<double>::memcpyDeviceToDevice(0,res,a,size*4*sizeof(double));
}

// vEqBF

void MechanicalObjectOpenCLVec3f_vEqBF(size_t size, _device_pointer res, const _device_pointer b, float f)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vEqBF");
    BARRIER(res,__FILE__,__LINE__);
    ERROR_OFFSET(res);
    ERROR_OFFSET(b)

//	size*=3;

    MechanicalObject_CreateProgramWithFloat();

    MechanicalObjectOpenCLVec3f_vEqBF_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3f_vEqBF_kernel->setArg<cl_mem>(1,&(b.m));
    MechanicalObjectOpenCLVec3f_vEqBF_kernel->setArg<float>(2,&f);

    const int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3f_vEqBF_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
    BARRIER(res,__FILE__,__LINE__);
}

// vPEqBF

void MechanicalObjectOpenCLVec3f_vPEqBF(size_t size, _device_pointer res, const _device_pointer b, float f)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vPEqBF");
    ERROR_OFFSET(res);
    ERROR_OFFSET(b);
    BARRIER(res,__FILE__,__LINE__);

//	size*=3;

    MechanicalObject_CreateProgramWithFloat();



    MechanicalObjectOpenCLVec3f_vPEqBF_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3f_vPEqBF_kernel->setArg<cl_mem>(1,&(b.m));
    MechanicalObjectOpenCLVec3f_vPEqBF_kernel->setArg<float>(2,&f);

    const int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3f_vPEqBF_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
    BARRIER(res,__FILE__,__LINE__);
}


// vDot (1/4)
#define RED_SIZE 512

void MechanicalObjectOpenCLVec3f_vDot(size_t size, float* res, const _device_pointer a, const _device_pointer b, _device_pointer tmp, float* cputmp)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vDot");
    BARRIER(a,__FILE__,__LINE__);
    ERROR_OFFSET(a);
    ERROR_OFFSET(b);
    ERROR_OFFSET(tmp);

    size*=3;
    MechanicalObject_CreateProgramWithFloat();

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
    BARRIER(a,__FILE__,__LINE__);
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


void MechanicalObjectOpenCLVec3f_vAdd(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vAdd");
    BARRIER(res,__FILE__,__LINE__);
    ERROR_OFFSET(res);
    ERROR_OFFSET(a);
    ERROR_OFFSET(b);

//	size*=3;

    MechanicalObject_CreateProgramWithFloat();

    MechanicalObjectOpenCLVec3f_vAdd_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3f_vAdd_kernel->setArg<cl_mem>(1,&(a.m));
    MechanicalObjectOpenCLVec3f_vAdd_kernel->setArg<cl_mem>(2,&(b.m));


    const int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3f_vAdd_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
    BARRIER(res,__FILE__,__LINE__);
}


void MechanicalObjectOpenCLVec3f_vPEq(size_t size, _device_pointer res, const _device_pointer a)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vPEq") ;
    BARRIER(res,__FILE__,__LINE__);
    ERROR_OFFSET(res);
    ERROR_OFFSET(a);

//	size*=3;



    MechanicalObject_CreateProgramWithFloat();

    MechanicalObjectOpenCLVec3f_vPEq_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3f_vPEq_kernel->setArg<cl_mem>(1,&(a.m));

    const int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3f_vPEq_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0*/
    BARRIER(res,__FILE__,__LINE__);
}



void MechanicalObjectOpenCLVec3f_vPEqBF2(size_t size, _device_pointer res1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer b2, float f2)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vPEqBF2");
    BARRIER(res1,__FILE__,__LINE__);
    ERROR_OFFSET(res1);
    ERROR_OFFSET(b1);
    ERROR_OFFSET(res2);
    ERROR_OFFSET(b2);

    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;


//	size*=3;

    MechanicalObject_CreateProgramWithFloat();




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

    BARRIER(res1,__FILE__,__LINE__);
}

void MechanicalObjectOpenCLVec3f_vIntegrate(size_t size, const _device_pointer a, _device_pointer v, _device_pointer x, float f_v_v, float f_v_a, float f_x_x, float f_x_v)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vIntegrate");
    BARRIER(a,__FILE__,__LINE__);
    ERROR_OFFSET(a);
    ERROR_OFFSET(v);
    ERROR_OFFSET(x);

    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;


//	size*=3;
    MechanicalObject_CreateProgramWithFloat();

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
    BARRIER(a,__FILE__,__LINE__);
}



void MechanicalObjectOpenCLVec3f_vOp2(size_t size, _device_pointer res1, const _device_pointer a1, const _device_pointer b1, float f1, _device_pointer res2, const _device_pointer a2, const _device_pointer b2, float f2)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vOp2");
    BARRIER(res1,__FILE__,__LINE__);
    ERROR_OFFSET(res1);
    ERROR_OFFSET(a1);
    ERROR_OFFSET(b2);
    ERROR_OFFSET(res2);
    ERROR_OFFSET(b1);
    ERROR_OFFSET(b2);

    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;


//	size*=3;
    MechanicalObject_CreateProgramWithFloat();

    MechanicalObjectOpenCLVec3f_vOp2_kernel->setArg<cl_mem>(0,&(res1.m));
    MechanicalObjectOpenCLVec3f_vOp2_kernel->setArg<cl_mem>(1,&(a1.m));
    MechanicalObjectOpenCLVec3f_vOp2_kernel->setArg<cl_mem>(2,&(b1.m));
    MechanicalObjectOpenCLVec3f_vOp2_kernel->setArg<float>(3,&f1);
    MechanicalObjectOpenCLVec3f_vOp2_kernel->setArg<cl_mem>(4,&res2.m);
    MechanicalObjectOpenCLVec3f_vOp2_kernel->setArg<cl_mem>(5,&a2.m);
    MechanicalObjectOpenCLVec3f_vOp2_kernel->setArg<cl_mem>(6,&b2.m);
    MechanicalObjectOpenCLVec3f_vOp2_kernel->setArg<float>(7,&f2);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);
    MechanicalObjectOpenCLVec3f_vOp2_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

    BARRIER(res1,__FILE__,__LINE__);
    DEBUG_TEXT("~MechanicalObjectOpenCLVec3f_vOp2");
}

//MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel
void MechanicalObjectOpenCLVec3f_vPEq4BF2(size_t size, _device_pointer res1, const _device_pointer b11, float f11, const _device_pointer b12, float f12, const _device_pointer b13, float f13, const _device_pointer b14, float f14,
        _device_pointer res2, const _device_pointer b21, float f21, const _device_pointer b22, float f22, const _device_pointer b23, float f23, const _device_pointer b24, float f24)
{
    DEBUG_TEXT("MechanicalObjectOpenCLVec3f_vPEq4BF2");
    BARRIER(res1,__FILE__,__LINE__);
    ERROR_OFFSET(res1);
    ERROR_OFFSET(b11);
    ERROR_OFFSET(b12);
    ERROR_OFFSET(b13);
    ERROR_OFFSET(b14);
    ERROR_OFFSET(res2);
    ERROR_OFFSET(b21);
    ERROR_OFFSET(b22);
    ERROR_OFFSET(b23);
    ERROR_OFFSET(b24);

    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;

    size*=3;
    MechanicalObject_CreateProgramWithFloat();

    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<cl_mem>(0,&(res1.m));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<cl_mem>(1,&(b11.m));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<float>(2,&(f11));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<cl_mem>(3,&(b12.m));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<float>(4,&(f12));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<cl_mem>(5,&(b13.m));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<float>(6,&f13);
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<cl_mem>(7,&(b14.m));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<float>(8,&f14);
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<cl_mem>(9,&(res2.m));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<cl_mem>(10,&(b21.m));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<float>(11,&(f21));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<cl_mem>(12,&(b22.m));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<float>(13,&(f22));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<cl_mem>(14,&(b23.m));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<float>(15,&f23);
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<cl_mem>(16,&(b24.m));
    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->setArg<float>(17,&f24);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);


    MechanicalObjectOpenCLVec3f_vPEq4BF2_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

    BARRIER(res1,__FILE__,__LINE__);
}








/////////////////////////
///////////////////////
////////////////////

OpenCLProgram* MechanicalObjectOpenCLDouble_program = NULL;
void MechanicalObject_CreateProgramWithDouble()
{

    if(MechanicalObjectOpenCLDouble_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="double";
        types["Real4"]="double4";

        MechanicalObjectOpenCLDouble_program
            = new OpenCLProgram("OpenCLMechanicalObject.cl",stringBSIZE,&types);

        MechanicalObjectOpenCLDouble_program->buildProgram();
    }
}



OpenCLKernel * MechanicalObjectOpenCLVec3d_vOp_kernel = NULL;
void MechanicalObjectOpenCLVec3d_vOp(size_t size, _device_pointer res, const _device_pointer a, const _device_pointer b, double f)
{
    NOT_IMPLEMENTED();
    DEBUG_TEXT( "MechanicalObjectOpenCLVec3d_vOp\t");
    BARRIER(res,__FILE__,__LINE__);
    ERROR_OFFSET(res)
    ERROR_OFFSET(a)
    ERROR_OFFSET(b)


    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;

    MechanicalObject_CreateProgramWithDouble();
    if(MechanicalObjectOpenCLVec3d_vOp_kernel==NULL)MechanicalObjectOpenCLVec3d_vOp_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLDouble_program,"MechanicalObject_Vec3t_vOp");
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(0,&res);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(1,&a);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<_device_pointer>(2,&b);
    MechanicalObjectOpenCLVec3f_vOp_kernel->setArg<double>(3,&f);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3d_vOp_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
    BARRIER(res,__FILE__,__LINE__);
}


OpenCLKernel * MechanicalObjectOpenCLVec3d_vMEq_kernel = NULL;
void MechanicalObjectOpenCLVec3d_vMEq(size_t size, _device_pointer res, double f)
{
    NOT_IMPLEMENTED();
    DEBUG_TEXT("MechanicalObjectOpenCLVec3d_vMEq");
    BARRIER(res,__FILE__,__LINE__);
    ERROR_OFFSET(res)

    size*=3;
    MechanicalObject_CreateProgramWithDouble();
    if(MechanicalObjectOpenCLVec3d_vMEq_kernel==NULL)MechanicalObjectOpenCLVec3d_vMEq_kernel
            = new OpenCLKernel(MechanicalObjectOpenCLDouble_program,"Vec1t_vMEq");
    MechanicalObjectOpenCLVec3d_vMEq_kernel->setArg<cl_mem>(0,&(res.m));
    MechanicalObjectOpenCLVec3d_vMEq_kernel->setArg<double>(1,&f);

    const int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    MechanicalObjectOpenCLVec3d_vMEq_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
    BARRIER(res,__FILE__,__LINE__);
}













void MechanicalObjectOpenCLVec3f1_vMEq(size_t /*size*/, _device_pointer /*res*/, float /*f*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3f1_vEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, float /*f*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3f1_vPEq(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3f1_vPEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, float /*f*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3f1_vAdd(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3f1_vOp(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/, float /*f*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3f1_vIntegrate(size_t /*size*/, const _device_pointer /*a*/, _device_pointer /*v*/, _device_pointer /*x*/, float /*f_v_v*/, float /*f_v_a*/, float /*f_x_x*/, float /*f_x_v*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3f1_vPEqBF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b1*/, float /*f1*/, _device_pointer /*res2*/, const _device_pointer /*b2*/, float /*f2*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3f1_vPEq4BF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b11*/, float /*f11*/, const _device_pointer /*b12*/, float /*f12*/, const _device_pointer /*b13*/, float /*f13*/, const _device_pointer /*b14*/, float /*f14*/,
        _device_pointer /*res2*/, const _device_pointer /*b21*/, float /*f21*/, const _device_pointer /*b22*/, float /*f22*/, const _device_pointer /*b23*/, float /*f23*/, const _device_pointer /*b24*/, float /*f24*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3f1_vOp2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*a1*/, const _device_pointer /*b1*/, float /*f1*/, _device_pointer /*res2*/, const _device_pointer /*a2*/, const _device_pointer /*b2*/, float /*f2*/) {NOT_IMPLEMENTED();}
int MechanicalObjectOpenCLVec3f1_vDotTmpSize(size_t /*size*/) {NOT_IMPLEMENTED(); return 0;}
void MechanicalObjectOpenCLVec3f1_vDot(size_t /*size*/, float* /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/, _device_pointer /*tmp*/, float*  /*cputmp*/) {NOT_IMPLEMENTED();}





void MechanicalObjectOpenCLVec3d_vEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, double /*f*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d_vPEq(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d_vPEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, double /*f*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d_vAdd(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d_vIntegrate(size_t /*size*/, const _device_pointer /*a*/, _device_pointer /*v*/, _device_pointer /*x*/, double /*f_v_v*/, double /*f_v_a*/, double /*f_x_x*/, double /*f_x_v*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d_vPEqBF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b1*/, double /*f1*/, _device_pointer /*res2*/, const _device_pointer /*b2*/, double /*f2*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d_vPEq4BF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b11*/, double /*f11*/, const _device_pointer /*b12*/, double /*f12*/, const _device_pointer /*b13*/, double /*f13*/, const _device_pointer /*b14*/, double /*f14*/,
        _device_pointer /*res2*/, const _device_pointer /*b21*/, double /*f21*/, const _device_pointer /*b22*/, double /*f22*/, const _device_pointer /*b23*/, double /*f23*/, const _device_pointer /*b24*/, double /*f24*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d_vOp2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*a1*/, const _device_pointer /*b1*/, double /*f1*/, _device_pointer /*res2*/, const _device_pointer /*a2*/, const _device_pointer /*b2*/, double /*f2*/) {NOT_IMPLEMENTED();}
int MechanicalObjectOpenCLVec3d_vDotTmpSize(size_t /*size*/) {NOT_IMPLEMENTED(); return 0;}
void MechanicalObjectOpenCLVec3d_vDot(size_t /*size*/, double* /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/, _device_pointer /*tmp*/, double*  /*cputmp*/) {NOT_IMPLEMENTED();}

void MechanicalObjectOpenCLVec3d1_vMEq(size_t /*size*/, _device_pointer /*res*/, double /*f*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d1_vEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, double /*f*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d1_vPEq(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d1_vPEqBF(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*b*/, double /*f*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d1_vAdd(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d1_vOp(size_t /*size*/, _device_pointer /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/, double /*f*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d1_vIntegrate(size_t /*size*/, const _device_pointer /*a*/, _device_pointer /*v*/, _device_pointer /*x*/, double /*f_v_v*/, double /*f_v_a*/, double /*f_x_x*/, double /*f_x_v*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d1_vPEqBF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b1*/, double /*f1*/, _device_pointer /*res2*/, const _device_pointer /*b2*/, double /*f2*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d1_vPEq4BF2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*b11*/, double /*f11*/, const _device_pointer /*b12*/, double /*f12*/, const _device_pointer /*b13*/, double /*f13*/, const _device_pointer /*b14*/, double /*f14*/,
        _device_pointer /*res2*/, const _device_pointer /*b21*/, double /*f21*/, const _device_pointer /*b22*/, double /*f22*/, const _device_pointer /*b23*/, double /*f23*/, const _device_pointer /*b24*/, double /*f24*/) {NOT_IMPLEMENTED();}
void MechanicalObjectOpenCLVec3d1_vOp2(size_t /*size*/, _device_pointer /*res1*/, const _device_pointer /*a1*/, const _device_pointer /*b1*/, double /*f1*/, _device_pointer /*res2*/, const _device_pointer /*a2*/, const _device_pointer /*b2*/, double /*f2*/) {NOT_IMPLEMENTED();}
int MechanicalObjectOpenCLVec3d1_vDotTmpSize(size_t /*size*/) {NOT_IMPLEMENTED(); return 0;}
void MechanicalObjectOpenCLVec3d1_vDot(size_t /*size*/, double* /*res*/, const _device_pointer /*a*/, const _device_pointer /*b*/, _device_pointer /*tmp*/, double*  /*cputmp*/) {NOT_IMPLEMENTED();}



























} // namespace OpenCL

} // namespace gpu

} // namespace sofa


#undef DEBUG_TEXT
