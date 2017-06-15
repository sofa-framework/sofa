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
#include "OpenCLSpringForceField.inl"

#include <SofaObjectInteraction/BoxStiffSpringForceField.inl>
#include <SofaDeformable/SpringForceField.inl>
#include <SofaDeformable/StiffSpringForceField.inl>
#include <SofaDeformable/MeshSpringForceField.inl>
#include <SofaDeformable/TriangleBendingSprings.inl>
#include <SofaDeformable/QuadBendingSprings.inl>
#include <sofa/core/behavior/PairInteractionForceField.inl>


#include <sofa/core/ObjectFactory.h>

#include "OpenCLProgram.h"
#include "OpenCLKernel.h"

#define DEBUG_TEXT(t) //printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);

namespace sofa
{

namespace core
{

namespace behavior
{

template class PairInteractionForceField<sofa::gpu::opencl::OpenCLVec3fTypes>;
template class PairInteractionForceField<sofa::gpu::opencl::OpenCLVec3f1Types>;
template class PairInteractionForceField<sofa::gpu::opencl::OpenCLVec3dTypes>;
template class PairInteractionForceField<sofa::gpu::opencl::OpenCLVec3d1Types>;

} // namespace behavior

} // namespace core

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

OpenCLProgram* SpringForceFieldOpenCLFloat_program = NULL;
OpenCLProgram* SpringForceFieldOpenCLDouble_program = NULL;


void SpringForceField_CreateProgramWithFloat()
{
    if(SpringForceFieldOpenCLFloat_program==NULL)
    {

        std::map<std::string, std::string> types;
        types["Real"]="float";
        types["Real4"]="float4";

        SpringForceFieldOpenCLFloat_program
            = new OpenCLProgram("OpenCLSpringForceField.cl",stringBSIZE,&types);

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
            = new OpenCLProgram("OpenCLSpringForceField.cl",stringBSIZE,&types);

        SpringForceFieldOpenCLDouble_program->buildProgram();

    }
}





OpenCLKernel * StiffSpringForceFieldOpenCL3f_addForce_kernel = NULL;
void StiffSpringForceFieldOpenCL3f_addForce(unsigned int size, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v, _device_pointer dfdx)
{

    int BSIZE = component::interactionforcefield::SpringForceFieldInternalData<sofa::gpu::opencl::OpenCLVec3fTypes>::BSIZE;
    DEBUG_TEXT("StiffSpringForceFieldOpenCL3f_addForce");
    SpringForceField_CreateProgramWithFloat();
    if(StiffSpringForceFieldOpenCL3f_addForce_kernel==NULL)StiffSpringForceFieldOpenCL3f_addForce_kernel
            = new OpenCLKernel(SpringForceFieldOpenCLFloat_program,"StiffSpringForceField3t_addForce");


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


OpenCLKernel * StiffSpringForceFieldOpenCL3f_addDForce_kernel = NULL;
void StiffSpringForceFieldOpenCL3f_addDForce(unsigned int size, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer dx, const _device_pointer x, const _device_pointer dfdx, float factor)
{
    int BSIZE = component::interactionforcefield::SpringForceFieldInternalData<sofa::gpu::opencl::OpenCLVec3fTypes>::BSIZE;
    DEBUG_TEXT("StiffSpringForceFieldOpenCL3f_addDForce");


    SpringForceField_CreateProgramWithFloat();
    if(StiffSpringForceFieldOpenCL3f_addDForce_kernel==NULL)StiffSpringForceFieldOpenCL3f_addDForce_kernel
            = new OpenCLKernel(SpringForceFieldOpenCLFloat_program,"StiffSpringForceField3t_addDForce");


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


void SpringForceFieldOpenCL3f_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/)
{
    std::cerr << "SpringForceFieldOpenCL3f_addForce not implemented" << std::endl; exit(703);
}
void SpringForceFieldOpenCL3f_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, unsigned int /*offset1*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, unsigned int /*offset2*/, const _device_pointer/*x2*/, const _device_pointer/*v2*/)
{
    std::cerr << "SpringForceFieldOpenCL3f_addExternalForce not implemented" << std::endl; exit(703);
}



OpenCLKernel * StiffSpringForceFieldOpenCL3f_addExternalForce_kernel = NULL;
void StiffSpringForceFieldOpenCL3f_addExternalForce(unsigned int size, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2, _device_pointer dfdx)
{
    //std::cerr << "StiffSpringForceFieldOpenCL3f_addExternalForce not implemented" << std::endl; exit(703);

    int BSIZE = component::interactionforcefield::SpringForceFieldInternalData<sofa::gpu::opencl::OpenCLVec3fTypes>::BSIZE;
    DEBUG_TEXT("StiffSpringForceFieldOpenCL3f_addExternalForce");
    SpringForceField_CreateProgramWithFloat();
    if(StiffSpringForceFieldOpenCL3f_addExternalForce_kernel==NULL)StiffSpringForceFieldOpenCL3f_addExternalForce_kernel
            = new OpenCLKernel(SpringForceFieldOpenCLFloat_program,"StiffSpringForceField3t_addExternalForce");


    StiffSpringForceFieldOpenCL3f_addExternalForce_kernel->setArg<unsigned int>   (0,&nbSpringPerVertex);
    StiffSpringForceFieldOpenCL3f_addExternalForce_kernel->setArg<_device_pointer>(1,&springs);
    StiffSpringForceFieldOpenCL3f_addExternalForce_kernel->setArg<unsigned int>   (2,&offset1);
    StiffSpringForceFieldOpenCL3f_addExternalForce_kernel->setArg<_device_pointer>(3,&f1);
    StiffSpringForceFieldOpenCL3f_addExternalForce_kernel->setArg<_device_pointer>(4,&x1);
    StiffSpringForceFieldOpenCL3f_addExternalForce_kernel->setArg<_device_pointer>(5,&v1);
    StiffSpringForceFieldOpenCL3f_addExternalForce_kernel->setArg<unsigned int>   (6,&offset2);
    StiffSpringForceFieldOpenCL3f_addExternalForce_kernel->setArg<_device_pointer>(7,&x2);
    StiffSpringForceFieldOpenCL3f_addExternalForce_kernel->setArg<_device_pointer>(8,&v2);
    StiffSpringForceFieldOpenCL3f_addExternalForce_kernel->setArg<_device_pointer>(9,&dfdx);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    StiffSpringForceFieldOpenCL3f_addExternalForce_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0
}

OpenCLKernel * StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel = NULL;
void StiffSpringForceFieldOpenCL3f_addExternalDForce(unsigned int size, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer dx1, const _device_pointer x1, unsigned int offset2, const _device_pointer dx2, const _device_pointer x2, const _device_pointer dfdx, float factor)
{
    //std::cerr << "StiffSpringForceFieldOpenCL3f_addExternalDForce not implemented" << std::endl; //exit(703);
    int BSIZE = component::interactionforcefield::SpringForceFieldInternalData<sofa::gpu::opencl::OpenCLVec3fTypes>::BSIZE;
    DEBUG_TEXT("StiffSpringForceFieldOpenCL3f_addExternalDForce");


    SpringForceField_CreateProgramWithFloat();
    if(StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel==NULL)StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel
            = new OpenCLKernel(SpringForceFieldOpenCLFloat_program,"StiffSpringForceField3t_addExternalDForce");


    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->setArg<unsigned int>   (0,&nbSpringPerVertex);
    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->setArg<_device_pointer>(1,&springs);
    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->setArg<unsigned int>   (2,&offset1);
    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->setArg<_device_pointer>(3,&f1);
    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->setArg<_device_pointer>(4,&dx1);
    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->setArg<_device_pointer>(5,&x1);
    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->setArg<unsigned int>   (6,&offset2);
    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->setArg<_device_pointer>(7,&dx2);
    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->setArg<_device_pointer>(8,&x2);
    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->setArg<_device_pointer>(9,&dfdx);
    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->setArg<float>          (10,&factor);

    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=((size%BSIZE)==0)?size:BSIZE*(size/BSIZE+1);

    //std::cout << __LINE__ << __FILE__ << " " << size << " " << nbSpringPerVertex << " " << springs.offset << " " << f.offset << " " <<  x.offset << " " << " " << dfdx.offset << "\n";
    //std::cout << local_size[0] << " " << size << " " <<work_size[0] << "\n";

    StiffSpringForceFieldOpenCL3f_addExternalDForce_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

}


void SpringForceFieldOpenCL3f1_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/)
{
    std::cerr << "SpringForceFieldOpenCL3f1_addForce not implemented" << std::endl; exit(703);
}
void SpringForceFieldOpenCL3f1_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, unsigned int /*offset1*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, unsigned int /*offset2*/, const _device_pointer/*x2*/, const _device_pointer/*v2*/)
{
    std::cerr << "SpringForceFieldOpenCL3f1_addExternalForce not implemented" << std::endl; exit(703);
}

void StiffSpringForceFieldOpenCL3f1_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/, _device_pointer/*dfdx*/)
{
    std::cerr << "StiffSpringForceFieldOpenCL3f1_addForce not implemented" << std::endl; exit(703);
}
void StiffSpringForceFieldOpenCL3f1_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, unsigned int /*offset1*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, unsigned int /*offset2*/, const _device_pointer/*x2*/, const _device_pointer/*v2*/, _device_pointer/*dfdx*/)
{
    std::cerr << "StiffSpringForceFieldOpenCL3f1_addExternalForce not implemented" << std::endl; exit(703);
}
void StiffSpringForceFieldOpenCL3f1_addDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*dx*/, const _device_pointer/*x*/, const _device_pointer/*dfdx*/, double/*factor*/)
{
    std::cerr << "StiffSpringForceFieldOpenCL3f1_addDForce not implemented" << std::endl; exit(703);
}
void StiffSpringForceFieldOpenCL3f1_addExternalDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, unsigned int /*offset1*/, _device_pointer/*f1*/, const _device_pointer/*dx1*/, const _device_pointer/*x1*/, unsigned int /*offset2*/, const _device_pointer/*dx2*/, const _device_pointer/*x2*/, const _device_pointer/*dfdx*/, double/*factor*/)
{
    std::cerr << "StiffSpringForceFieldOpenCL3f1_addExternalDForce not implemented" << std::endl; exit(703);
}




void SpringForceFieldOpenCL3d_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/) {DEBUG_TEXT("no implemented");}
void SpringForceFieldOpenCL3d_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, unsigned int /*offset1*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, unsigned int /*offset2*/, const _device_pointer/*x2*/, const _device_pointer/*v2*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/, _device_pointer/*dfdx*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, unsigned int /*offset1*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, unsigned int /*offset2*/, const _device_pointer/*x2*/, const _device_pointer/*v2*/, _device_pointer/*dfdx*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d_addDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*dx*/, const _device_pointer/*x*/, const _device_pointer/*dfdx*/, double/*factor*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d_addExternalDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, unsigned int /*offset1*/, _device_pointer/*f1*/, const _device_pointer/*dx1*/, const _device_pointer/*x1*/, unsigned int /*offset2*/, const _device_pointer/*dx2*/, const _device_pointer/*x2*/, const _device_pointer/*dfdx*/, double/*factor*/) {DEBUG_TEXT("no implemented");}

void SpringForceFieldOpenCL3d1_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/) {DEBUG_TEXT("no implemented");}
void SpringForceFieldOpenCL3d1_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, unsigned int /*offset1*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, unsigned int /*offset2*/, const _device_pointer/*x2*/, const _device_pointer/*v2*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d1_addForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*x*/, const _device_pointer/*v*/, _device_pointer/*dfdx*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d1_addExternalForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, unsigned int /*offset1*/, _device_pointer/*f1*/, const _device_pointer/*x1*/, const _device_pointer/*v1*/, unsigned int /*offset1*/, const _device_pointer/*x2*/, const _device_pointer/*v2*/, _device_pointer/*dfdx*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d1_addDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, _device_pointer/*f*/, const _device_pointer/*dx*/, const _device_pointer/*x*/, const _device_pointer/*dfdx*/, double/*factor*/) {DEBUG_TEXT("no implemented");}
void StiffSpringForceFieldOpenCL3d1_addExternalDForce(unsigned int /*nbVertex*/, unsigned int /*nbSpringPerVertex*/, const _device_pointer /*springs*/, unsigned int /*offset1*/, _device_pointer/*f1*/, const _device_pointer/*dx1*/, const _device_pointer/*x1*/, unsigned int /*offset2*/, const _device_pointer/*dx2*/, const _device_pointer/*x2*/, const _device_pointer/*dfdx*/, double/*factor*/) {DEBUG_TEXT("no implemented");}

} // namespace opencl

} // namespace gpu

} // namespace sofa
