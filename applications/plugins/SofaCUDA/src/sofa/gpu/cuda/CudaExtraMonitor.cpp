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
#include "CudaTypes.h"

#include <SofaValidation/ExtraMonitor.h>
#include <SofaValidation/ExtraMonitor.inl>
#include <SofaValidation/Monitor.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::misc
{

template class SOFA_GPU_CUDA_API ExtraMonitor<gpu::cuda::CudaRigid3fTypes>;
template class SOFA_GPU_CUDA_API ExtraMonitor<gpu::cuda::CudaVec6fTypes>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API ExtraMonitor<gpu::cuda::CudaRigid3dTypes>;
template class SOFA_GPU_CUDA_API ExtraMonitor<gpu::cuda::CudaVec6dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE

}// namespace sofa::component::misc


namespace sofa::gpu::cuda
{
int ExtraMonitorCudaClass = core::RegisterObject("Supports GPU-side computation using CUDA")
        .add< component::misc::ExtraMonitor<CudaRigid3fTypes> >()
        .add< component::misc::ExtraMonitor<CudaVec6fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::misc::ExtraMonitor<CudaRigid3dTypes> >()
        .add< component::misc::ExtraMonitor<CudaVec6dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

}// namespace sofa::gpu::cuda
