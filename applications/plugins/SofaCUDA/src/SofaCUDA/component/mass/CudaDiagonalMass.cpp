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
#ifndef SOFA_GPU_CUDA_CUDADIAGONALMASS_CPP
#define SOFA_GPU_CUDA_CUDADIAGONALMASS_CPP

#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaCUDA/component/mass/CudaDiagonalMass.inl>
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/topology/container/dynamic/EdgeSetGeometryAlgorithms.inl>
#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.inl>
#include <sofa/component/topology/container/dynamic/TetrahedronSetGeometryAlgorithms.inl>
#include <sofa/component/topology/container/dynamic/HexahedronSetGeometryAlgorithms.inl>
#include <sofa/component/topology/container/dynamic/QuadSetGeometryAlgorithms.inl>

namespace sofa
{

namespace component
{

namespace mass
{

template class SOFA_GPU_CUDA_API DiagonalMass<sofa::gpu::cuda::CudaVec3fTypes>;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API DiagonalMass<sofa::gpu::cuda::CudaVec3dTypes>;
#endif

} // namespace mass

} // namespace component


namespace gpu
{

namespace cuda
{


// Register in the Factory
int DiagonalMassCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
    .add< component::mass::DiagonalMass<CudaVec3fTypes> >()

#ifdef SOFA_GPU_CUDA_DOUBLE
    .add< component::mass::DiagonalMass<CudaVec3dTypes> >()
 // SOFA_GPU_CUDA_DOUBLE
#endif
        ;


} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif // SOFA_GPU_CUDA_CUDADIAGONALMASS_CPP
