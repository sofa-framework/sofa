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
#ifndef SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_H
#define SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_H

#include "CudaTypes.h"
#include <SofaBaseMechanics/IdentityMapping.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::gpu::cuda;

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT(const core::ConstraintParams * cparams, Data<InMatrixDeriv>& dOut, const Data<MatrixDeriv>& dIn);


//////// CudaVec3f1

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJT( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );





#ifndef SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_CPP

using namespace sofa::defaulttype;
using namespace sofa::component::mapping;
using namespace sofa::gpu::cuda;

extern template class  IdentityMapping< CudaVec3fTypes, CudaVec3fTypes>;
#ifndef SOFA_DOUBLE
extern template class  IdentityMapping< CudaVec3fTypes, Vec3fTypes>;
extern template class  IdentityMapping< Vec3fTypes, CudaVec3fTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class  IdentityMapping< CudaVec3fTypes, Vec3dTypes>;
extern template class  IdentityMapping< Vec3dTypes, CudaVec3fTypes>;
#endif

#ifdef SOFA_GPU_CUDA_DOUBLE
extern template class  IdentityMapping< CudaVec3fTypes, CudaVec3dTypes>;
extern template class  IdentityMapping< CudaVec3dTypes, CudaVec3fTypes>;
extern template class  IdentityMapping< CudaVec3dTypes, CudaVec3dTypes>;
extern template class  IdentityMapping< CudaVec3dTypes, Vec3fTypes>;
extern template class  IdentityMapping< CudaVec3dTypes, Vec3dTypes>;
#ifndef SOFA_DOUBLE
extern template class  IdentityMapping< Vec3dTypes, CudaVec3dTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class  IdentityMapping< Vec3fTypes, CudaVec3dTypes>;
#endif

extern template class  IdentityMapping< CudaVec3d1Types, ExtVec3dTypes >;
extern template class  IdentityMapping< CudaVec3dTypes, ExtVec3dTypes >;
#endif
extern template class  IdentityMapping< CudaVec3f1Types, ExtVec3fTypes >;
extern template class  IdentityMapping< CudaVec3f1Types, CudaVec3f1Types>;
extern template class  IdentityMapping< CudaVec3f1Types, Vec3fTypes>;
#ifndef SOFA_FLOAT
extern template class  IdentityMapping< Vec3dTypes, CudaVec3f1Types>;
extern template class  IdentityMapping< CudaVec3f1Types, Vec3dTypes>;
extern template class  IdentityMapping< CudaVec3f1Types, ExtVec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class  IdentityMapping< Vec3fTypes, ExtVec3fTypes>;
#endif
extern template class  IdentityMapping< CudaVec3f1Types, CudaVec3fTypes>;
extern template class  IdentityMapping< CudaVec3fTypes, CudaVec3f1Types>;

#endif


} // namespace mapping

} // namespace component

} // namespace sofa

#endif
