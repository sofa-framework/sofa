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
#include "OpenCLSPHFluidForceField.inl"
#include <sofa/core/ObjectFactory.h>

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



void SPHFluidForceFieldOpenCL3f_computeDensity(unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3f* params,_device_pointer pos4, const _device_pointer x) {NOT_IMPLEMENTED();}
void SPHFluidForceFieldOpenCL3f_addForce (unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3f* params,_device_pointer f, const _device_pointer pos4, const _device_pointer v) {NOT_IMPLEMENTED();}
void SPHFluidForceFieldOpenCL3f_addDForce(unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3f* params,_device_pointer f, const _device_pointer pos4, const _device_pointer v, const _device_pointer dx) {NOT_IMPLEMENTED();}



void SPHFluidForceFieldOpenCL3d_computeDensity(unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3d* params,_device_pointer pos4, const _device_pointer x) {NOT_IMPLEMENTED();}
void SPHFluidForceFieldOpenCL3d_addForce (unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3d* params,_device_pointer f, const _device_pointer pos4, const _device_pointer v) {NOT_IMPLEMENTED();}
void SPHFluidForceFieldOpenCL3d_addDForce(unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3d* params,_device_pointer f, const _device_pointer pos4, const _device_pointer v, const _device_pointer dx) {NOT_IMPLEMENTED();}




} // namespace OpenCL

} // namespace gpu

} // namespace sofa
