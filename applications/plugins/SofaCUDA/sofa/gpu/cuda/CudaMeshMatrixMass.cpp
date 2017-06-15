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
#include <sofa/gpu/cuda/CudaMeshMatrixMass.inl>
#include <SofaMiscForceField/MeshMatrixMass.inl>
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/behavior/ForceField.inl>

#include <SofaBaseTopology/PointSetGeometryAlgorithms.inl>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.inl>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.inl>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.inl>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.inl>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>
#include "CudaTypes.h"

namespace sofa
{

namespace component
{

namespace mass
{

template class MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes, float>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template class MeshMatrixMass<sofa::gpu::cuda::CudaVec2dTypes, double>;
#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace mass

} // namespace component

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaMeshMatrixMassClass)

int MeshMatrixMassClassCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::mass::MeshMatrixMass<CudaVec2fTypes, float > >(true)
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::mass::MeshMatrixMass<CudaVec2dTypes, double > >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa

