/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <SofaCUDA/config.h>
#include <sofa/component/mechanicalload/ConstantForceField.h>

#if !defined(SOFA_COMPONENT_FORCEFIELD_CUDACONSTANTFORCEFIELD_CPP)

#include <sofa/gpu/cuda/CudaTypes.h>

namespace sofa::component::mechanicalload
{

using namespace sofa::gpu::cuda;

extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec3Types>;
extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec2Types>;
extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec1Types>;
extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec6Types>;
extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaRigid3Types>;
extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaRigid2Types>;

#ifdef SOFA_GPU_CUDA_DOUBLE
extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec3dTypes>;
extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec2dTypes>;
extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec1dTypes>;
extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec6dTypes>;
extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaRigid3dTypes>;
extern template class SOFA_GPU_CUDA_API ConstantForceField<CudaRigid2dTypes>;
#endif

} // namespace sofa::component::mechanicalload
#endif



