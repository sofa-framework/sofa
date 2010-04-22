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
#include "OpenCLUniformMass.inl"
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
        .add< component::mass::UniformMass<OpenCLVec3fTypes,float> >()
        .add< component::mass::UniformMass<OpenCLVec3f1Types,float> >()
        .add< component::mass::UniformMass<OpenCLRigid3fTypes,sofa::defaulttype::Rigid3fMass> >()
        .add< component::mass::UniformMass<OpenCLVec3dTypes,double> >()
        .add< component::mass::UniformMass<OpenCLVec3d1Types,double> >()
        .add< component::mass::UniformMass<OpenCLRigid3dTypes,sofa::defaulttype::Rigid3dMass> >()
        ;







///////////////////////////////////////
//             kernels

sofa::helper::OpenCLProgram* UniformMassOpenCLFloat_program;
sofa::helper::OpenCLProgram* UniformMassOpenCLDouble_program;


void UniformMass_CreateProgramWithFloat()
{
    if(UniformMassOpenCLFloat_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="float";
        types["Real4"]="float4";

        UniformMassOpenCLFloat_program
            = new sofa::helper::OpenCLProgram(sofa::helper::OpenCLProgram::loadSource("OpenCLUniformMass.cl"),&types);

        UniformMassOpenCLFloat_program->buildProgram();
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
        std::cout << UniformMassOpenCLFloat_program->buildLog(0);
    }
}

void UniformMass_CreateProgramWithDouble()
{

    if(UniformMassOpenCLDouble_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="double";
        types["Real4"]="double4";

        UniformMassOpenCLDouble_program
            = new sofa::helper::OpenCLProgram(sofa::helper::OpenCLProgram::loadSource("OpenCLUniformMass.cl"),&types);

        UniformMassOpenCLDouble_program->buildProgram();

    }
}




sofa::helper::OpenCLKernel * UniformMassOpenCL3f_addForce_kernel;
void UniformMassOpenCL3f_addForce(unsigned int size, const float* mg, _device_pointer f)
{
    DEBUG_TEXT("UniformMassOpenCL3f_addForce");

    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    UniformMass_CreateProgramWithFloat();
    if(UniformMassOpenCL3f_addForce_kernel==NULL)UniformMassOpenCL3f_addForce_kernel
            = new sofa::helper::OpenCLKernel(UniformMassOpenCLFloat_program,"UniformMass_addForce_v2");

    UniformMassOpenCL3f_addForce_kernel->setArg<unsigned int>(0,&size);
    UniformMassOpenCL3f_addForce_kernel->setArg<float>(1,&mg[0]);
    UniformMassOpenCL3f_addForce_kernel->setArg<float>(2,&mg[1]);
    UniformMassOpenCL3f_addForce_kernel->setArg<float>(3,&mg[2]);
    UniformMassOpenCL3f_addForce_kernel->setArg<_device_pointer>(4,&f);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    //std::cout << __LINE__ << __FILE__ << " " << size << " " << nbSpringPerVertex << " " << springs.offset << " " << f.offset << " " <<  x.offset << " " <<  v.offset<< " " << dfdx.offset << "\n";
    //std::cout << local_size[0] << " " << size << " " <<work_size[0] << "\n";

    UniformMassOpenCL3f_addForce_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

}

sofa::helper::OpenCLKernel * UniformMassOpenCL3f_addMDX_kernel;
void UniformMassOpenCL3f_addMDx(unsigned int size, float mass, _device_pointer res, const _device_pointer dx)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    DEBUG_TEXT("UniformMassOpenCL3f_addMDx");

    UniformMass_CreateProgramWithFloat();
    if(UniformMassOpenCL3f_addMDX_kernel==NULL)UniformMassOpenCL3f_addMDX_kernel
            = new sofa::helper::OpenCLKernel(UniformMassOpenCLFloat_program,"UniformMass_addMDx");


    UniformMassOpenCL3f_addMDX_kernel->setArg<float>(0,&mass);
    UniformMassOpenCL3f_addMDX_kernel->setArg<_device_pointer>(1,&res);
    UniformMassOpenCL3f_addMDX_kernel->setArg<_device_pointer>(2,&dx);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    //std::cout << __LINE__ << __FILE__ << " " << size << " " <<res.offset << " " << dx.offset << "\n";
    //std::cout << local_size[0] << " " << size << " " <<work_size[0] << "\n";

    UniformMassOpenCL3f_addMDX_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

}


void UniformMassOpenCL3f_accFromF(unsigned int /*size*/, float /*mass*/, _device_pointer /*a*/, const _device_pointer /*f*/) {DEBUG_TEXT("no implemented"); exit(0);}

void UniformMassOpenCL3f1_addMDx(unsigned int /*size*/, float /*mass*/, _device_pointer /*res*/, const _device_pointer /*dx*/) {DEBUG_TEXT("no implemented"); exit(0);}
void UniformMassOpenCL3f1_accFromF(unsigned int /*size*/, float /*mass*/, _device_pointer /*a*/, const _device_pointer /*f*/) {DEBUG_TEXT("no implemented"); exit(0);}
void UniformMassOpenCL3f1_addForce(unsigned int /*size*/, const float* /*mg*/, _device_pointer /*f*/) {DEBUG_TEXT("no implemented"); exit(0);}

void UniformMassOpenCL3d_addMDx(unsigned int /*size*/, double /*mass*/, _device_pointer /*res*/, const _device_pointer /*dx*/) {DEBUG_TEXT("no implemented"); exit(0);}
void UniformMassOpenCL3d_accFromF(unsigned int /*size*/, double /*mass*/, _device_pointer /*a*/, const _device_pointer /*f*/) {DEBUG_TEXT("no implemented"); exit(0);}
void UniformMassOpenCL3d_addForce(unsigned int /*size*/, const double* /*mg*/, _device_pointer /*f*/) {DEBUG_TEXT("no implemented"); exit(0);}

void UniformMassOpenCL3d1_addMDx(unsigned int /*size*/, double /*mass*/, _device_pointer /*res*/, const _device_pointer /*dx*/) {DEBUG_TEXT("no implemented"); exit(0);}
void UniformMassOpenCL3d1_accFromF(unsigned int /*size*/, double /*mass*/, _device_pointer /*a*/, const _device_pointer /*f*/) {DEBUG_TEXT("no implemented"); exit(0);}
void UniformMassOpenCL3d1_addForce(unsigned int /*size*/, const double* /*mg*/, _device_pointer /*f*/) {DEBUG_TEXT("no implemented"); exit(0);}














} // namespace opencl

} // namespace gpu

} // namespace sofa

#undef BSIZE
