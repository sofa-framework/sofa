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
#include "OpenCLUniformMass.inl"
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/behavior/Mass.inl>
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

SOFA_DECL_CLASS(OpenCLUniformMass)

int UniformMassOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
#ifndef SOFA_DOUBLE
        .add< component::mass::UniformMass<OpenCLVec3fTypes,float> >()
        .add< component::mass::UniformMass<OpenCLVec3f1Types,float> >()
        .add< component::mass::UniformMass<OpenCLRigid3fTypes,sofa::defaulttype::Rigid3fMass> >()
#endif
#ifndef SOFA_FLOAT
        .add< component::mass::UniformMass<OpenCLVec3dTypes,double> >()
        .add< component::mass::UniformMass<OpenCLVec3d1Types,double> >()
        .add< component::mass::UniformMass<OpenCLRigid3dTypes,sofa::defaulttype::Rigid3dMass> >()
#endif
        ;







///////////////////////////////////////
//             kernels

OpenCLProgram* UniformMassOpenCLFloat_program = NULL;

OpenCLKernel * UniformMassOpenCL3f_addForce_kernel = NULL;
OpenCLKernel * UniformMassOpenCL3f_addMDX_kernel = NULL;
OpenCLKernel * UniformMassOpenCL3f_accFromF_kernel = NULL;
void UniformMass_CreateProgramWithFloat()
{
    if(UniformMassOpenCLFloat_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="float";
        types["Real4"]="float4";

        UniformMassOpenCLFloat_program
            = new OpenCLProgram("OpenCLUniformMass.cl",stringBSIZE,&types);

        UniformMassOpenCLFloat_program->buildProgram();
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
        std::cout << UniformMassOpenCLFloat_program->buildLog(0);

        UniformMassOpenCL3f_addForce_kernel
            = new OpenCLKernel(UniformMassOpenCLFloat_program,"UniformMass_addForce_v2");

        UniformMassOpenCL3f_addMDX_kernel
            = new OpenCLKernel(UniformMassOpenCLFloat_program,"UniformMass_addMDx");

        UniformMassOpenCL3f_accFromF_kernel
            = new OpenCLKernel(UniformMassOpenCLFloat_program,"UniformMass_accFromF");

    }
}


void UniformMassOpenCL3f_addForce(unsigned int size, const float* mg, _device_pointer f)
{
    DEBUG_TEXT("UniformMassOpenCL3f_addForce");
    ERROR_OFFSET(f);

    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    UniformMass_CreateProgramWithFloat();

    UniformMassOpenCL3f_addForce_kernel->setArg<unsigned int>(0,&size);
    UniformMassOpenCL3f_addForce_kernel->setArg<float>(1,&mg[0]);
    UniformMassOpenCL3f_addForce_kernel->setArg<float>(2,&mg[1]);
    UniformMassOpenCL3f_addForce_kernel->setArg<float>(3,&mg[2]);
    UniformMassOpenCL3f_addForce_kernel->setArg<_device_pointer>(4,&f);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    UniformMassOpenCL3f_addForce_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

    DEBUG_TEXT("~UniformMassOpenCL3f_addForce");
}

void UniformMassOpenCL3f_addMDx(unsigned int size, float mass, _device_pointer res, const _device_pointer dx)
{
    DEBUG_TEXT("UniformMassOpenCL3f_addMDx");
    ERROR_OFFSET(res)
    ERROR_OFFSET(dx)
//	size*=3;

    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;


    UniformMass_CreateProgramWithFloat();

    UniformMassOpenCL3f_addMDX_kernel->setArg<float>(0,&mass);
    UniformMassOpenCL3f_addMDX_kernel->setArg<_device_pointer>(1,&res);
    UniformMassOpenCL3f_addMDX_kernel->setArg<_device_pointer>(2,&dx);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    UniformMassOpenCL3f_addMDX_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
    DEBUG_TEXT("~UniformMassOpenCL3f_addMDx");
}

void UniformMassOpenCL3f_accFromF(unsigned int size, float mass, _device_pointer a, const _device_pointer f)
{

    DEBUG_TEXT("UniformMassOpenCL3f_accFromF");
    ERROR_OFFSET(a)
    ERROR_OFFSET(f)

    size*=3;

    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    float inv_mass = 1.0/mass;

    UniformMass_CreateProgramWithFloat();

    UniformMassOpenCL3f_accFromF_kernel->setArg<float>(0,&inv_mass);
    UniformMassOpenCL3f_accFromF_kernel->setArg<_device_pointer>(1,&a);
    UniformMassOpenCL3f_accFromF_kernel->setArg<_device_pointer>(2,&f);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    UniformMassOpenCL3f_accFromF_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0




    DEBUG_TEXT("~UniformMassOpenCL3f_accFromF");
}

void UniformMassOpenCL3f1_addMDx(unsigned int /*size*/, float /*mass*/, _device_pointer /*res*/, const _device_pointer /*dx*/) {NOT_IMPLEMENTED();}
void UniformMassOpenCL3f1_accFromF(unsigned int /*size*/, float /*mass*/, _device_pointer /*a*/, const _device_pointer /*f*/) {NOT_IMPLEMENTED();}
void UniformMassOpenCL3f1_addForce(unsigned int /*size*/, const float* /*mg*/, _device_pointer /*f*/) {NOT_IMPLEMENTED();}

void UniformMassOpenCL3d_addMDx(unsigned int /*size*/, double /*mass*/, _device_pointer /*res*/, const _device_pointer /*dx*/) {NOT_IMPLEMENTED();}
void UniformMassOpenCL3d_accFromF(unsigned int /*size*/, double /*mass*/, _device_pointer /*a*/, const _device_pointer /*f*/) {NOT_IMPLEMENTED();}
void UniformMassOpenCL3d_addForce(unsigned int /*size*/, const double* /*mg*/, _device_pointer /*f*/) {NOT_IMPLEMENTED();}

void UniformMassOpenCL3d1_addMDx(unsigned int /*size*/, double /*mass*/, _device_pointer /*res*/, const _device_pointer /*dx*/) {NOT_IMPLEMENTED();}
void UniformMassOpenCL3d1_accFromF(unsigned int /*size*/, double /*mass*/, _device_pointer /*a*/, const _device_pointer /*f*/) {NOT_IMPLEMENTED();}
void UniformMassOpenCL3d1_addForce(unsigned int /*size*/, const double* /*mg*/, _device_pointer /*f*/) {NOT_IMPLEMENTED();}














} // namespace opencl

} // namespace gpu

} // namespace sofa

#undef BSIZE
