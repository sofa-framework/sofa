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
#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaCUDA/component/mapping/linear/CudaBarycentricMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::mapping::linear
{

using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::gpu::cuda;

// Spread the instanciations over multiple files for more efficient and lightweight compilation

// instanciations involving both CudaVec3f1Types and Vec3dTypes



template class SOFA_GPU_CUDA_API BarycentricMapping< Vec3Types, CudaVec3f1Types>;
template class SOFA_GPU_CUDA_API BarycentricMapping< CudaVec3f1Types, Vec3Types>;


} // namespace sofa::component::mapping::linear

namespace sofa::gpu::cuda
{

using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::component::mapping::linear;

int BarycentricMappingCudaClass_3f1_d = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< BarycentricMapping< Vec3Types, CudaVec3f1Types> >()
        .add< BarycentricMapping< CudaVec3f1Types, Vec3Types> >()

        ;

} // namespace sofa::gpu::cuda
