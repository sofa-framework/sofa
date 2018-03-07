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
#include "OpenCLSPHFluidForceField.inl"
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/ObjectFactory.h>

#define DEBUG_TEXT(t) //printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);


namespace sofa
{

namespace gpu
{

namespace opencl
{

SOFA_DECL_CLASS(OpenCLSPHFluidForceField)

int SPHFluidForceFieldOpenCLClass = core::RegisterObject("Supports GPU-side computations using OpenCL")
        .add< component::forcefield::SPHFluidForceField< OpenCLVec3fTypes > >()
        .add< component::forcefield::SPHFluidForceField< OpenCLVec3dTypes > >()
        ;



/////////////////////////////

OpenCLProgram* SPHFluidForceFieldOpenCLFloat_program = NULL;


void SPHFluidForceField_CreateProgramWithFloat()
{
    if(SPHFluidForceFieldOpenCLFloat_program==NULL)
    {
        std::map<std::string, std::string> types;
        types["Real"]="float";
        types["Real4"]="float4";

        SPHFluidForceFieldOpenCLFloat_program
            = new OpenCLProgram("OpenCLSPHFluidForceField.cl",stringBSIZE,&types);

        SPHFluidForceFieldOpenCLFloat_program->buildProgram();

        std::cout << SPHFluidForceFieldOpenCLFloat_program->buildLog(0);
        sofa::gpu::opencl::myopenclShowError(__FILE__,__LINE__);
    }
}

OpenCLKernel *SPHFluidForceFieldOpenCL3f_computeDensity_kernel = NULL;
void SPHFluidForceFieldOpenCL3f_computeDensity(unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3f* params,_device_pointer pos4, const _device_pointer x)
{

    DEBUG_TEXT("SPHFluidForceFieldOpenCL3f_computeDensity");
    BARRIER(cells,__FILE__,__LINE__);

    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    SPHFluidForceField_CreateProgramWithFloat();
//std::cout << BSIZE << " " << stringBSIZE << "\n";
    if(SPHFluidForceFieldOpenCL3f_computeDensity_kernel==NULL)SPHFluidForceFieldOpenCL3f_computeDensity_kernel
            = new OpenCLKernel(SPHFluidForceFieldOpenCLFloat_program,"SPHFluidForceField_computeDensity");


    SPHFluidForceFieldOpenCL3f_computeDensity_kernel->setArg<unsigned int>(0,&size);
    SPHFluidForceFieldOpenCL3f_computeDensity_kernel->setArg<_device_pointer>(1,&cells);
    SPHFluidForceFieldOpenCL3f_computeDensity_kernel->setArg<_device_pointer>(2,&cellGhost);
    SPHFluidForceFieldOpenCL3f_computeDensity_kernel->setArg<GPUSPHFluid3f>(3,params);
    SPHFluidForceFieldOpenCL3f_computeDensity_kernel->setArg<_device_pointer>(4,&pos4);
    SPHFluidForceFieldOpenCL3f_computeDensity_kernel->setArg<_device_pointer>(5,&x);



    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=60*BSIZE;

//	std::cout << "COMPUTE DENSITY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    //std::cout << __LINE__ << __FILE__ << " " << size << " " << nbSpringPerVertex << " " << springs.offset << " " << f.offset << " " <<  x.offset << " " <<  v.offset<< " " << dfdx.offset << "\n";
    //std::cout << local_size[0] << " " << size << " " <<work_size[0] << "\n";

    SPHFluidForceFieldOpenCL3f_computeDensity_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

    DEBUG_TEXT("~SPHFluidForceFieldOpenCL3f_computeDensity");
    BARRIER(cells,__FILE__,__LINE__);

}

OpenCLKernel *SPHFluidForceFieldOpenCL3f_addForce_kernel = NULL;
void SPHFluidForceFieldOpenCL3f_addForce (unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3f* params,_device_pointer f, const _device_pointer pos4, const _device_pointer v)
{



    DEBUG_TEXT("SPHFluidForceFieldOpenCL3f_addForce");

    BARRIER(cells,__FILE__,__LINE__);

    /*std::cout <<
    	params->h<<" "<<         ///< particles radius
    	params->h2<<" "<<       ///< particles radius squared
    	params->stiffness<<" "<< ///< pressure stiffness
    	params->mass<<" "<<      ///< particles mass
    	params->mass2<<" "<<     ///< particles mass squared
    	params->density0<<" "<<  ///< 1000 kg/m3 for water
    	params->viscosity<<" "<<
    	params->surfaceTension<<" "<<

    	// Precomputed constants for smoothing kernels
    	params->CWd<<" "<<          ///< = constWd(h)
    	params->CgradWd<<" "<<      ///< = constGradWd(h)
    	params->CgradWp<<" "<<      ///< = constGradWp(h)
    	params->ClaplacianWv<<" "<< ///< = constLaplacianWv(h)
    	params->CgradWc<<" "<<      ///< = constGradWc(h)
    	params->ClaplacianWc<<"\n"; ///< = constLaplacianWc(h)
    */



    int BSIZE = gpu::opencl::OpenCLMemoryManager<float>::BSIZE;
    SPHFluidForceField_CreateProgramWithFloat();

    if(SPHFluidForceFieldOpenCL3f_addForce_kernel==NULL)SPHFluidForceFieldOpenCL3f_addForce_kernel
            = new OpenCLKernel(SPHFluidForceFieldOpenCLFloat_program,"SPHFluidForceField_addForce");



    SPHFluidForceFieldOpenCL3f_addForce_kernel->setArg<unsigned int>(0,&size);
    SPHFluidForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(1,&cells);
    SPHFluidForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(2,&cellGhost);
    SPHFluidForceFieldOpenCL3f_addForce_kernel->setArg<GPUSPHFluid3f>(3,params);
    SPHFluidForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(4,&f);
    SPHFluidForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(5,&pos4);
    SPHFluidForceFieldOpenCL3f_addForce_kernel->setArg<_device_pointer>(6,&v);



    size_t local_size[1];
    local_size[0]=BSIZE;

    size_t work_size[1];
    work_size[0]=60*BSIZE;

    //std::cout << __LINE__ << __FILE__ << " " << size << " " << nbSpringPerVertex << " " << springs.offset << " " << f.offset << " " <<  x.offset << " " <<  v.offset<< " " << dfdx.offset << "\n";
    //std::cout << local_size[0] << " " << size << " " <<work_size[0] << "\n";

    SPHFluidForceFieldOpenCL3f_addForce_kernel->execute(0,1,NULL,work_size,local_size);	//note: num_device = const = 0

    DEBUG_TEXT("~SPHFluidForceFieldOpenCL3f_addForce");
    BARRIER(cells,__FILE__,__LINE__);
}

void SPHFluidForceFieldOpenCL3f_addDForce(unsigned int /*size*/, const _device_pointer /*cells*/, const _device_pointer /*cellGhost*/, GPUSPHFluid3f* /*params*/,_device_pointer /*f*/, const _device_pointer /*pos4*/, const _device_pointer /*v*/, const _device_pointer /*dx*/) {NOT_IMPLEMENTED();}



void SPHFluidForceFieldOpenCL3d_computeDensity(unsigned int /*size*/, const _device_pointer /*cells*/, const _device_pointer /*cellGhost*/, GPUSPHFluid3d* /*params*/,_device_pointer /*pos4*/, const _device_pointer /*x*/) {NOT_IMPLEMENTED();}
void SPHFluidForceFieldOpenCL3d_addForce (unsigned int /*size*/, const _device_pointer /*cells*/, const _device_pointer /*cellGhost*/, GPUSPHFluid3d* /*params*/,_device_pointer /*f*/, const _device_pointer /*pos4*/, const _device_pointer /*v*/) {NOT_IMPLEMENTED();}
void SPHFluidForceFieldOpenCL3d_addDForce(unsigned int /*size*/, const _device_pointer /*cells*/, const _device_pointer /*cellGhost*/, GPUSPHFluid3d* /*params*/,_device_pointer /*f*/, const _device_pointer /*pos4*/, const _device_pointer /*v*/, const _device_pointer /*dx*/) {NOT_IMPLEMENTED();}

} // namespace OpenCL

} // namespace gpu

} // namespace sofa
