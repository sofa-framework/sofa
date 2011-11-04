/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include "OpenCLSpringForceField.inl"
#include <sofa/component/interactionforcefield/BoxStiffSpringForceField.inl>
#include <sofa/core/ObjectFactory.h>

#include "OpenCLProgram.h"
#include "OpenCLKernel.h"

#define DEBUG_TEXT(t) //printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template class SpringForceField<sofa::gpu::opencl::OpenCLVec3fTypes>;
template class StiffSpringForceField<sofa::gpu::opencl::OpenCLVec3fTypes>;
template class MeshSpringForceField<sofa::gpu::opencl::OpenCLVec3fTypes>;
template class BoxStiffSpringForceField<gpu::opencl::OpenCLVec3fTypes>;

template class SpringForceField<sofa::gpu::opencl::OpenCLVec3f1Types>;
template class StiffSpringForceField<sofa::gpu::opencl::OpenCLVec3f1Types>;
template class MeshSpringForceField<sofa::gpu::opencl::OpenCLVec3f1Types>;
template class BoxStiffSpringForceField<gpu::opencl::OpenCLVec3f1Types>;

template class SpringForceField<sofa::gpu::opencl::OpenCLVec3dTypes>;
template class StiffSpringForceField<sofa::gpu::opencl::OpenCLVec3dTypes>;
template class MeshSpringForceField<sofa::gpu::opencl::OpenCLVec3dTypes>;
template class BoxStiffSpringForceField<gpu::opencl::OpenCLVec3dTypes>;

template class SpringForceField<sofa::gpu::opencl::OpenCLVec3d1Types>;
template class StiffSpringForceField<sofa::gpu::opencl::OpenCLVec3d1Types>;
template class MeshSpringForceField<sofa::gpu::opencl::OpenCLVec3d1Types>;
template class BoxStiffSpringForceField<gpu::opencl::OpenCLVec3d1Types>;

} // namespace interactionforcefield

} // namespace component

namespace gpu
{

namespace opencl
{

SOFA_DECL_CLASS(OpenCLSpringForceField)
SOFA_DECL_CLASS(OpenCLBoxStiffSpringForceField)

int SpringForceFieldOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< component::interactionforcefield::SpringForceField<OpenCLVec3fTypes> >()
        .add< component::interactionforcefield::SpringForceField<OpenCLVec3f1Types> >()
        .add< component::interactionforcefield::SpringForceField<OpenCLVec3dTypes> >()
        .add< component::interactionforcefield::SpringForceField<OpenCLVec3d1Types> >()
        ;

int StiffSpringForceFieldOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< component::interactionforcefield::StiffSpringForceField<OpenCLVec3fTypes> >()
        .add< component::interactionforcefield::StiffSpringForceField<OpenCLVec3f1Types> >()
        .add< component::interactionforcefield::StiffSpringForceField<OpenCLVec3dTypes> >()
        .add< component::interactionforcefield::StiffSpringForceField<OpenCLVec3d1Types> >()
        ;

int MeshSpringForceFieldOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< component::interactionforcefield::MeshSpringForceField<OpenCLVec3fTypes> >()
        .add< component::interactionforcefield::MeshSpringForceField<OpenCLVec3f1Types> >()
        .add< component::interactionforcefield::MeshSpringForceField<OpenCLVec3dTypes> >()
        .add< component::interactionforcefield::MeshSpringForceField<OpenCLVec3d1Types> >()
        ;

int TriangleBendingSpringsOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< component::interactionforcefield::TriangleBendingSprings<OpenCLVec3fTypes> >()
        .add< component::interactionforcefield::TriangleBendingSprings<OpenCLVec3f1Types> >()
        .add< component::interactionforcefield::TriangleBendingSprings<OpenCLVec3dTypes> >()
        .add< component::interactionforcefield::TriangleBendingSprings<OpenCLVec3d1Types> >()
        ;

int QuadBendingSpringsOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< component::interactionforcefield::QuadBendingSprings<OpenCLVec3fTypes> >()
        .add< component::interactionforcefield::QuadBendingSprings<OpenCLVec3f1Types> >()
        .add< component::interactionforcefield::QuadBendingSprings<OpenCLVec3dTypes> >()
        .add< component::interactionforcefield::QuadBendingSprings<OpenCLVec3d1Types> >()
        ;

int BoxStiffSpringForceFieldOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< component::interactionforcefield::BoxStiffSpringForceField<OpenCLVec3fTypes> >()
        .add< component::interactionforcefield::BoxStiffSpringForceField<OpenCLVec3f1Types> >()
        .add< component::interactionforcefield::BoxStiffSpringForceField<OpenCLVec3dTypes> >()
        .add< component::interactionforcefield::BoxStiffSpringForceField<OpenCLVec3d1Types> >()
        ;








/////////////////////////////////////////////////////////////////////////
//            kernels

sofa::helper::OpenCLProgram* SpringForceFieldOpenCLFloat_program;
sofa::helper::OpenCLProgram* SpringForceFieldOpenCLDouble_program;


void SpringForceField_CreateProgramWithFloat()
{
    if(SpringForceFieldOpenCLFloat_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="float";
        types["Real4"]="float4";

        std::string source =*sofa::helper::OpenCLProgram::loadSource("OpenCLSpringForceField.cl");

        SpringForceFieldOpenCLFloat_program
            = new sofa::helper::OpenCLProgram(&source,&types);

        SpringForceFieldOpenCLFloat_program->buildProgram();
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
        std::cout << SpringForceFieldOpenCLFloat_program->buildLog(0);
    }
}

void SpringForceField_CreateProgramWithDouble()
{

    if(SpringForceFieldOpenCLDouble_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="double";
        types["Real4"]="double4";

        SpringForceFieldOpenCLDouble_program
            = new sofa::helper::OpenCLProgram(sofa::helper::OpenCLProgram::loadSource("OpenCLSpringForceField.cl"),&types);

        SpringForceFieldOpenCLDouble_program->buildProgram();

    }
}





sofa::helper::OpenCLKernel * StiffSpringForceFieldOpenCL3f_addForce_kernel;
void StiffSpringForceFieldOpenCL3f_addForce(unsigned int size, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v, _device_pointer dfdx)
{

    int BSIZE = component::interactionforcefield::SpringForceFieldInternalData<sofa::gpu::opencl::OpenCLVec3fTypes>::BSIZE;
    DEBUG_TEXT("StiffSpringForceFieldOpenCL3f_addForce");
    SpringForceField_CreateProgramWithFloat();
    if(StiffSpringForceFieldOpenCL3f_addForce_kernel==NULL)StiffSpringForceFieldOpenCL3f_addForce_kernel
            = new sofa::helper::OpenCLKernel(SpringForceFieldOpenCLFloat_program,"StiffSpringForceField3t_addForce");


    StiffSpringForceFieldOpenCL3f_addForce_kernel->setArg<unsigned int>(0,&nbSpringPerVertex);
    StiffSpringForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(1,&springs);
    StiffSpringForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(2,&f);
    StiffSpringForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(3,&x);
    StiffSpringForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(4,&v);
    StiffSpringForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(5,&dfdx);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    StiffSpringForceFieldOpenCL3f_addForce_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

}


sofa::helper::OpenCLKernel * StiffSpringForceFieldOpenCL3f_addDForce_kernel;
void StiffSpringForceFieldOpenCL3f_addDForce(unsigned int size, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer dx, const _device_pointer x, const _device_pointer dfdx, float factor)
{
    int BSIZE = component::interactionforcefield::SpringForceFieldInternalData<sofa::gpu::opencl::OpenCLVec3fTypes>::BSIZE;
    DEBUG_TEXT("StiffSpringForceFieldOpenCL3f_addDForce");


//	dim3 threads(BSIZE,1);
//	dim3 grid((size+BSIZE-1)/BSIZE,1);
#ifdef USE_TEXTURE
    CudaSpringForceFieldInputTextures<float,CudaVec3<float> >::setX((const CudaVec3<float>*)x);
    CudaSpringForceFieldInputTextures<float,CudaVec3<float> >::setDX((const CudaVec3<float>*)dx);
    size_t sizeBuffer = BSIZE*3*sizeof(float);
#else
//	size_t sizeBuffer = BSIZE*6*sizeof(float);
#endif

    SpringForceField_CreateProgramWithFloat();
    if(StiffSpringForceFieldOpenCL3f_addDForce_kernel==NULL)StiffSpringForceFieldOpenCL3f_addDForce_kernel
            = new sofa::helper::OpenCLKernel(SpringForceFieldOpenCLFloat_program,"StiffSpringForceField3t_addDForce");


    StiffSpringForceFieldOpenCL3f_addDForce_kernel->setArg<unsigned int>(0,&nbSpringPerVertex);
    StiffSpringForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(1,&springs);
    StiffSpringForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(2,&f);
    StiffSpringForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(3,&dx);
    StiffSpringForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(4,&x);
    StiffSpringForceFieldOpenCL3f_addDForce_kernel->setArg<_device_pointer>(5,&dfdx);
    StiffSpringForceFieldOpenCL3f_addDForce_kernel->setArg<float>(6,&factor);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    //std::cout << __LINE__ << __FILE__ << " " << size << " " << nbSpringPerVertex << " " << springs.offset << " " << f.offset << " " <<  x.offset << " " << " " << dfdx.offset << "\n";
    //std::cout << local_size[0] << " " << size << " " <<work_size[0] << "\n";

    StiffSpringForceFieldOpenCL3f_addDForce_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0



//	StiffSpringForceFieldCuda3t_addDForce_kernel<float><<< grid, threads,
//	>>>(nbSpringPerVertex, (const GPUSpring*)springs, (float*)f, (const float*)dx, (const float*)x, (const float*)dfdx, (float)factor);

}


void SpringForceFieldOpenCL3f_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/) {DEBUG_TEXT("no implemented"); exit(0);}
void SpringForceFieldOpenCL3f_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, const _device_pointer/*x1*/, const _device_pointer/*v2*/) {DEBUG_TEXT("no implemented"); exit(0);}
void StiffSpringForceFieldOpenCL3f_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, const _device_pointer/*x1*/, const _device_pointer/*v2*/, _device_pointer/*dfdx*/) {DEBUG_TEXT("no implemented"); exit(0);}
void StiffSpringForceFieldOpenCL3f_addExternalDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*dx1*/, const _device_pointer/*x1*/, const _device_pointer/*dx2*/, const _device_pointer/*x1*/, const _device_pointer/*dfdx*/, double/*factor*/) {DEBUG_TEXT("no implemented"); exit(0);}


void SpringForceFieldOpenCL3f1_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/) {DEBUG_TEXT("no implemented");}
void SpringForceFieldOpenCL3f1_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, const _device_pointer/*x1*/, const _device_pointer/*v2*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3f1_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/, _device_pointer/*dfdx*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3f1_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, const _device_pointer/*x1*/, const _device_pointer/*v2*/, _device_pointer/*dfdx*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3f1_addDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*dx*/, const _device_pointer/*x*/, const _device_pointer/*dfdx*/, double/*factor*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3f1_addExternalDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*dx1*/, const _device_pointer/*x1*/, const _device_pointer/*dx2*/, const _device_pointer/*x1*/, const _device_pointer/*dfdx*/, double/*factor*/) {DEBUG_TEXT("no implemented");}



void SpringForceFieldOpenCL3d_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/) {DEBUG_TEXT("no implemented");}
void SpringForceFieldOpenCL3d_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, const _device_pointer/*x1*/, const _device_pointer/*v2*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/, _device_pointer/*dfdx*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, const _device_pointer/*x1*/, const _device_pointer/*v2*/, _device_pointer/*dfdx*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d_addDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*dx*/, const _device_pointer/*x*/, const _device_pointer/*dfdx*/, double/*factor*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d_addExternalDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*dx1*/, const _device_pointer/*x1*/, const _device_pointer/*dx2*/, const _device_pointer/*x1*/, const _device_pointer/*dfdx*/, double/*factor*/) {DEBUG_TEXT("no implemented");}

void SpringForceFieldOpenCL3d1_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/) {DEBUG_TEXT("no implemented");}
void SpringForceFieldOpenCL3d1_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, const _device_pointer/*x1*/, const _device_pointer/*v2*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d1_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/, _device_pointer/*dfdx*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d1_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, const _device_pointer/*x1*/, const _device_pointer/*v2*/, _device_pointer/*dfdx*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d1_addDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*dx*/, const _device_pointer/*x*/, const _device_pointer/*dfdx*/, double/*factor*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d1_addExternalDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f1*/, const _device_pointer/*dx1*/, const _device_pointer/*x1*/, const _device_pointer/*dx2*/, const _device_pointer/*x1*/, const _device_pointer/*dfdx*/, double/*factor*/) {DEBUG_TEXT("no implemented");}

} // namespace opencl

} // namespace gpu

} // namespace sofa
