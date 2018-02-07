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
#include <sofa/gpu/cuda/CudaSpatialGridContainer.inl>
#include <SofaBaseMechanics/MechanicalObject.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace container
{

using namespace sofa::defaulttype;
using namespace sofa::gpu::cuda;
using namespace core::behavior;


SOFA_DECL_CLASS(CudaSpatialGridContainer)
SOFA_DECL_CLASS(CudaSpatialGrid)

int SpatialGridContainerCudaClass = core::RegisterObject("GPU support using CUDA.")
        .add< SpatialGridContainer<CudaVec3fTypes> >()
        ;

template class SOFA_GPU_CUDA_API SpatialGridContainer< CudaVec3fTypes >;
template class SOFA_GPU_CUDA_API SpatialGrid< SpatialGridTypes< CudaVec3fTypes > >;

#ifdef SOFA_GPU_CUDA_DOUBLE

template class SpatialGridContainer< CudaVec3dTypes >;
template class SpatialGrid< SpatialGridTypes< CudaVec3dTypes > >;

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace container

} // namespace component

} // namespace sofa
