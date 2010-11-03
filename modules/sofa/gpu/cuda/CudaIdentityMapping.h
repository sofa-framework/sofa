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
#ifndef SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_H
#define SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_H

#include "CudaTypes.h"
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <>
void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::apply( OutDataVecCoord& dOut, const InDataVecCoord& dIn, const core::MechanicalParams* mparams );

template <>
void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( OutDataVecDeriv& dOut, const InDataVecDeriv& dIn, const core::MechanicalParams* mparams );

template <>
void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( InDataVecDeriv& dOut, const OutDataVecDeriv& dIn, const core::MechanicalParams* mparams );

//////// CudaVec3f1

template <>
void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::apply( OutDataVecCoord& dOut, const InDataVecCoord& dIn, const core::MechanicalParams* mparams );

template <>
void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJ( OutDataVecDeriv& dOut, const InDataVecDeriv& dIn, const core::MechanicalParams* mparams );

template <>
void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJT( InDataVecDeriv& dOut, const OutDataVecDeriv& dIn, const core::MechanicalParams* mparams );

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
