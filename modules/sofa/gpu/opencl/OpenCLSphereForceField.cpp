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
#include "OpenCLSphereForceField.inl"
#include <sofa/core/ObjectFactory.h>

#include "OpenCLProgram.h"
#include "OpenCLKernel.h"

#define DEBUG_TEXT(t) //printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);

namespace sofa
{

namespace gpu
{

namespace opencl
{

SOFA_DECL_CLASS(OpenCLSphereForceField)

int SphereForceFieldOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< component::forcefield::SphereForceField<OpenCLVec3fTypes> >()
        .add< component::forcefield::SphereForceField<OpenCLVec3f1Types> >()
        .add< component::forcefield::SphereForceField<OpenCLVec3dTypes> >()
        .add< component::forcefield::SphereForceField<OpenCLVec3d1Types> >()
        ;





////////////////////////////////////////////////////////////////////////////////////
//start kernel

sofa::helper::OpenCLProgram* SphereForceFieldOpenCLFloat_program;
sofa::helper::OpenCLProgram* SphereForceFieldOpenCLDouble_program;


void SphereForceField_CreateProgramWithFloat()
{
    if(SphereForceFieldOpenCLFloat_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="float";
        types["Real4"]="float4";

        SphereForceFieldOpenCLFloat_program
            = new sofa::helper::OpenCLProgram(sofa::helper::OpenCLProgram::loadSource("OpenCLSphereForceField.cl"),&types);

        SphereForceFieldOpenCLFloat_program->buildProgram();
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
        std::cout << SphereForceFieldOpenCLFloat_program->buildLog(0);
        std::cout << SphereForceFieldOpenCLFloat_program->sourceLog();
    }
}

void SphereForceField_CreateProgramWithDouble()
{

    if(SphereForceFieldOpenCLDouble_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="double";
        types["Real4"]="double4";

        SphereForceFieldOpenCLDouble_program
            = new sofa::helper::OpenCLProgram(sofa::helper::OpenCLProgram::loadSource("OpenCLSphereForceField.cl"),&types);

        SphereForceFieldOpenCLDouble_program->buildProgram();

    }
}



typedef struct f4
{
    float a;
    float b;
    float c;
    float d;
    f4(float aa,float bb,float cc,float dd)
    {
        a=aa; b=bb; c=cc; d=dd;
    }
} float4;

sofa::helper::OpenCLKernel * SphereForceFieldOpenCL3f_addForce_kernel;
void SphereForceFieldOpenCL3f_addForce(unsigned int size, GPUSphere* sphere, _device_pointer penetration, _device_pointer f, const _device_pointer x, const _device_pointer v)
{
    DEBUG_TEXT( "SphereForceFieldOpenCL3f_addForce");
    float4 sc(sphere->center.x(),sphere->center.y(),sphere->center.z(),0.0);
    float4 sd(sphere->r ,sphere->stiffness,sphere->damping,0.0);

    SphereForceField_CreateProgramWithFloat();
    if(SphereForceFieldOpenCL3f_addForce_kernel==NULL)SphereForceFieldOpenCL3f_addForce_kernel
            = new sofa::helper::OpenCLKernel(SphereForceFieldOpenCLFloat_program,"addForce");


    SphereForceFieldOpenCL3f_addForce_kernel->setArg<float4>(0,&sc);
    SphereForceFieldOpenCL3f_addForce_kernel->setArg<float4>(1,&sd);
    SphereForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(2,&penetration);
    SphereForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(3,&f);
    SphereForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(4,&x);
    SphereForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(5,&v);


    size_t work_size[1];
    work_size[0]=size;

    SphereForceFieldOpenCL3f_addForce_kernel->execute(0,1,NULL,work_size,NULL);	//note: num_device = const = 0

}


sofa::helper::OpenCLKernel * SphereForceFieldOpenCL3f_addDForce_kernel;
void SphereForceFieldOpenCL3f_addDForce(unsigned int size, GPUSphere* sphere, const _device_pointer penetration, _device_pointer f, const _device_pointer dx)
{
    DEBUG_TEXT( "SphereForceFieldOpenCL3f_addDForce");
    float4 sc(sphere->center.x(),sphere->center.y(),sphere->center.z(),0.0);

    SphereForceField_CreateProgramWithFloat();
    if(SphereForceFieldOpenCL3f_addDForce_kernel==NULL)SphereForceFieldOpenCL3f_addDForce_kernel
            = new sofa::helper::OpenCLKernel(SphereForceFieldOpenCLFloat_program,"addDForce");


    SphereForceFieldOpenCL3f_addDForce_kernel->setArg<float4>(0,&sc);
    SphereForceFieldOpenCL3f_addDForce_kernel->setArg<float>(1,&(sphere->stiffness));
    SphereForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(2,&penetration);
    SphereForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(3,&f);
    SphereForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(4,&dx);


    size_t work_size[1];
    work_size[0]=size;

    SphereForceFieldOpenCL3f_addDForce_kernel->execute(0,1,NULL,work_size,NULL);	//note: num_device = const = 0

}



void SphereForceFieldOpenCL3f1_addForce(unsigned int /*size*/, GPUSphere* /*sphere*/, _device_pointer /*penetration*/, _device_pointer /*f*/, const _device_pointer /*x*/, const _device_pointer /*v*/) {DEBUG_TEXT("no implemented");}
void SphereForceFieldOpenCL3f1_addDForce(unsigned int /*size*/, GPUSphere* /*sphere*/, const _device_pointer /*penetration*/, _device_pointer /*f*/, const _device_pointer /*dx*/) {DEBUG_TEXT("no implemented");}
} // namespace opencl

} // namespace gpu

} // namespace sofa

#undef DEBUG_TEXT
