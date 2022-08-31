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
#include <sofa/component/mechanicalload/ConstantForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/gpu/cuda/CudaTypes.h>

namespace sofa::gpu::cuda
{
    using namespace sofa::component::mechanicalload;

    // Register in the Factory
    int ConstantForceFieldCudaClass = core::RegisterObject("Constant forces applied to given degrees of freedom")
        .add< ConstantForceField<CudaVec3Types> >()
        .add< ConstantForceField<CudaVec2Types> >()
        .add< ConstantForceField<CudaVec1Types> >()
        .add< ConstantForceField<CudaVec6Types> >()
        .add< ConstantForceField<CudaRigid3Types> >()
        .add< ConstantForceField<CudaRigid2Types> >()
    ;
} // namespace sofa::gpu::cuda

namespace sofa::component::mechanicalload
{
    using namespace sofa::gpu::cuda;

    template <> SOFA_GPU_CUDA_API
    SReal ConstantForceField<CudaRigid3Types>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const { return 0; }
    template <> SOFA_GPU_CUDA_API
    SReal ConstantForceField<CudaRigid2Types>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const { return 0; }

    template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec3Types>;
    template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec2Types>;
    template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec1Types>;
    template class SOFA_GPU_CUDA_API ConstantForceField<CudaVec6Types>;
    template class SOFA_GPU_CUDA_API ConstantForceField<CudaRigid3Types>;
    template class SOFA_GPU_CUDA_API ConstantForceField<CudaRigid2Types>;
} // namespace sofa::component::mechanicalload