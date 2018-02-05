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
#include "OpenCLTypes.h"
#include "OpenCLProgram.h"
#include "OpenCLKernel.h"
#include "OpenCLPlaneForceField.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/ForceField.inl>

#define DEBUG_TEXT(t) //printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);


namespace sofa
{

namespace gpu
{

namespace opencl
{

SOFA_DECL_CLASS(OpenCLPlaneForceField)

int PlaneForceFieldOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< component::forcefield::PlaneForceField<OpenCLVec3fTypes> >()
        .add< component::forcefield::PlaneForceField<OpenCLVec3f1Types> >()
        .add< component::forcefield::PlaneForceField<OpenCLVec3dTypes> >()
        .add< component::forcefield::PlaneForceField<OpenCLVec3d1Types> >()
        ;


////////////////////////////////////////////////////////////////////////////////////
//start kernel

OpenCLProgram* PlaneForceFieldOpenCLFloat_program = NULL;

OpenCLKernel * PlaneForceFieldOpenCL3f_addForce_kernel = NULL;
OpenCLKernel * PlaneForceFieldOpenCL3f_addDForce_kernel = NULL;

void PlaneForceField_CreateProgramWithFloat()
{
    if(PlaneForceFieldOpenCLFloat_program==NULL)
    {
        PlaneForceFieldOpenCLFloat_program
            = new OpenCLProgram();
        PlaneForceFieldOpenCLFloat_program->setSourceFile("OpenCLGenericParticleForceField.cl",stringBSIZE);
        std::string macros;
        if (OpenCLProgram::loadSource("OpenCLGenericParticleForceField_Plane.macrocl",&macros))
        {
            PlaneForceFieldOpenCLFloat_program->addMacros(&macros,"all");
            PlaneForceFieldOpenCLFloat_program->addMacros(&macros,"float");
        }
        PlaneForceFieldOpenCLFloat_program->createProgram();
        PlaneForceFieldOpenCLFloat_program->buildProgram();
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
        std::cout << PlaneForceFieldOpenCLFloat_program->buildLog(0);
        //	std::cout << PlaneForceFieldOpenCLFloat_program->sourceLog();


        //create kernels
        PlaneForceFieldOpenCL3f_addForce_kernel
            = new OpenCLKernel(PlaneForceFieldOpenCLFloat_program,"GenericParticleForceField_3f_addForce_Plane");

        PlaneForceFieldOpenCL3f_addDForce_kernel
            = new OpenCLKernel(PlaneForceFieldOpenCLFloat_program,"GenericParticleForceField_3f_addDForce_Plane");
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

void PlaneForceFieldOpenCL3f_addForce(unsigned int size, GPUPlane<float>* plane, _device_pointer penetration, _device_pointer f, const _device_pointer x, const _device_pointer v)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    DEBUG_TEXT( "PlaneForceFieldOpenCL3f_addForce");
    float4 pl(plane->normal.x(),plane->normal.y(),plane->normal.z(),0.0);
    float4 pd(plane->d ,plane->stiffness,plane->damping,0.0);

    PlaneForceField_CreateProgramWithFloat();

    PlaneForceFieldOpenCL3f_addForce_kernel->setArg<float4>(0,&pl);
    PlaneForceFieldOpenCL3f_addForce_kernel->setArg<float4>(1,&pd);
    PlaneForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(2,&penetration);
    PlaneForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(3,&f);
    PlaneForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(4,&x);
    PlaneForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(5,&v);


    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);


    PlaneForceFieldOpenCL3f_addForce_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

}


void PlaneForceFieldOpenCL3f_addDForce(unsigned int size, GPUPlane<float>* plane, const _device_pointer penetration, _device_pointer f, const _device_pointer dx)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    DEBUG_TEXT( "PlaneForceFieldOpenCL3f_addDForce");
    float4 pl(plane->normal.x(),plane->normal.y(),plane->normal.z(),0.0);

    PlaneForceField_CreateProgramWithFloat();



    PlaneForceFieldOpenCL3f_addDForce_kernel->setArg<float4>(0,&pl);
    PlaneForceFieldOpenCL3f_addDForce_kernel->setArg<float>(1,&(plane->stiffness));
    PlaneForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(2,&penetration);
    PlaneForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(3,&f);
    PlaneForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(4,&dx);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    PlaneForceFieldOpenCL3f_addDForce_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

}


void PlaneForceFieldOpenCL3f1_addForce(unsigned int /*size*/, GPUPlane<float>* /*plane*/, _device_pointer /*penetration*/, _device_pointer /*f*/, const _device_pointer /*x*/, const _device_pointer /*v*/) {NOT_IMPLEMENTED();}
void PlaneForceFieldOpenCL3f1_addDForce(unsigned int /*size*/, GPUPlane<float>* /*plane*/, const _device_pointer /*penetration*/, _device_pointer /*f*/, const _device_pointer /*dx*/) {NOT_IMPLEMENTED();}


void PlaneForceFieldOpenCL3d_addForce(unsigned int /*size*/, GPUPlane<double>* /*plane*/, _device_pointer /*penetration*/, _device_pointer /*f*/, const _device_pointer /*x*/, const _device_pointer /*v*/) {NOT_IMPLEMENTED();}
void PlaneForceFieldOpenCL3d_addDForce(unsigned int /*size*/, GPUPlane<double>* /*plane*/, const _device_pointer /*penetration*/, _device_pointer /*f*/, const _device_pointer /*dx*/) {NOT_IMPLEMENTED();}

void PlaneForceFieldOpenCL3d1_addForce(unsigned int /*size*/, GPUPlane<double>* /*plane*/, _device_pointer /*penetration*/, _device_pointer /*f*/, const _device_pointer /*x*/, const _device_pointer /*v*/) {NOT_IMPLEMENTED();}
void PlaneForceFieldOpenCL3d1_addDForce(unsigned int /*size*/, GPUPlane<double>* /*plane*/, const _device_pointer /*penetration*/, _device_pointer /*f*/, const _device_pointer /*dx*/) {NOT_IMPLEMENTED();}


} // namespace opencl

} // namespace gpu

} // namespace sofa

#undef DEBUG_TEXT
