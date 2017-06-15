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
#ifndef SOFAOPENCL_OPENCLIDENTITYMAPPING_H
#define SOFAOPENCL_OPENCLIDENTITYMAPPING_H

#include "OpenCLTypes.h"
#include <SofaBaseMechanics/IdentityMapping.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3fTypes, gpu::opencl::OpenCLVec3fTypes>::apply( const core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3fTypes, gpu::opencl::OpenCLVec3fTypes>::applyJ( const core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3fTypes, gpu::opencl::OpenCLVec3fTypes>::applyJT( const core::MechanicalParams* mparams /* PARAMS FIRST */, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );

//////// OpenCLVec3f1

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3f1Types, gpu::opencl::OpenCLVec3f1Types>::apply( const core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3f1Types, gpu::opencl::OpenCLVec3f1Types>::applyJ( const core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3f1Types, gpu::opencl::OpenCLVec3f1Types>::applyJT( const core::MechanicalParams* mparams /* PARAMS FIRST */, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
