/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include "OpenCLTypes.h"
#include "OpenCLSphereForceField.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/ForceField.inl>

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

OpenCLProgram* SphereForceFieldOpenCLFloat_program = NULL;

OpenCLKernel * SphereForceFieldOpenCL3f_addForce_kernel = NULL;
OpenCLKernel * SphereForceFieldOpenCL3f_addDForce_kernel = NULL;

/*
void SphereForceField_CreateProgramWithFloat()
{
	if(SphereForceFieldOpenCLFloat_program==NULL)
	{

		std::map<std::string, std::string> types;
		types["Real"]="float";
		types["Real4"]="float4";

		SphereForceFieldOpenCLFloat_program
				= new OpenCLProgram("OpenCLSphereForceField.cl",stringBSIZE,&types);

		SphereForceFieldOpenCLFloat_program->buildProgram();
		sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
		std::cout << SphereForceFieldOpenCLFloat_program->buildLog(0);
	}
}
*/

void SphereForceField_CreateProgramWithFloat()
{
    if(SphereForceFieldOpenCLFloat_program==NULL)
    {
        SphereForceFieldOpenCLFloat_program
            = new OpenCLProgram();

        SphereForceFieldOpenCLFloat_program->setSourceFile("OpenCLGenericParticleForceField.cl", stringBSIZE);
        std::string macros;
        if (OpenCLProgram::loadSource("OpenCLGenericParticleForceField_Sphere.macrocl",&macros))
        {
            SphereForceFieldOpenCLFloat_program->addMacros(&macros,"all");
            SphereForceFieldOpenCLFloat_program->addMacros(&macros,"float");
        }
        SphereForceFieldOpenCLFloat_program->createProgram();
        SphereForceFieldOpenCLFloat_program->buildProgram();
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
        std::cout << SphereForceFieldOpenCLFloat_program->buildLog(0);
        //	std::cout << SphereForceFieldOpenCLFloat_program->sourceLog();


        //create kernels
        SphereForceFieldOpenCL3f_addForce_kernel
            = new OpenCLKernel(SphereForceFieldOpenCLFloat_program,"GenericParticleForceField_3f_addForce_Sphere");

        SphereForceFieldOpenCL3f_addDForce_kernel
            = new OpenCLKernel(SphereForceFieldOpenCLFloat_program,"GenericParticleForceField_3f_addDForce_Sphere");
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


void SphereForceFieldOpenCL3f_addForce(unsigned int size, GPUSphere* sphere, _device_pointer penetration, _device_pointer f, const _device_pointer x, const _device_pointer v)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    DEBUG_TEXT( "SphereForceFieldOpenCL3f_addForce");
    BARRIER(f,__FILE__,__LINE__);
    float4 sc(sphere->center.x(),sphere->center.y(),sphere->center.z(),0.0);
    float4 sd(sphere->r ,sphere->stiffness,sphere->damping,0.0);

    SphereForceField_CreateProgramWithFloat();

    SphereForceFieldOpenCL3f_addForce_kernel->setArg<float4>(0,&sc);
    SphereForceFieldOpenCL3f_addForce_kernel->setArg<float4>(1,&sd);
    SphereForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(2,&penetration);
    SphereForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(3,&f);
    SphereForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(4,&x);
    SphereForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(5,&v);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    SphereForceFieldOpenCL3f_addForce_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

    BARRIER(f,__FILE__,__LINE__);
}


void SphereForceFieldOpenCL3f_addDForce(unsigned int size, GPUSphere* sphere, const _device_pointer penetration, _device_pointer f, const _device_pointer dx)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    DEBUG_TEXT( "SphereForceFieldOpenCL3f_addDForce");
    BARRIER(penetration,__FILE__,__LINE__);
    float4 sc(sphere->center.x(),sphere->center.y(),sphere->center.z(),0.0);

    SphereForceField_CreateProgramWithFloat();

    SphereForceFieldOpenCL3f_addDForce_kernel->setArg<float4>(0,&sc);
    SphereForceFieldOpenCL3f_addDForce_kernel->setArg<float>(1,&(sphere->stiffness));
    SphereForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(2,&penetration);
    SphereForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(3,&f);
    SphereForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(4,&dx);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    SphereForceFieldOpenCL3f_addDForce_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
    BARRIER(penetration,__FILE__,__LINE__);
}



void SphereForceFieldOpenCL3f1_addForce(unsigned int /*size*/, GPUSphere* /*sphere*/, _device_pointer /*penetration*/, _device_pointer /*f*/, const _device_pointer /*x*/, const _device_pointer /*v*/) {NOT_IMPLEMENTED();}
void SphereForceFieldOpenCL3f1_addDForce(unsigned int /*size*/, GPUSphere* /*sphere*/, const _device_pointer /*penetration*/, _device_pointer /*f*/, const _device_pointer /*dx*/) {NOT_IMPLEMENTED();}
} // namespace opencl

} // namespace gpu

} // namespace sofa

#undef DEBUG_TEXT
