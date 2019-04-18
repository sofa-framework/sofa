/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include "CudaTypes.h"
#include "CudaDiagonalMass.inl"
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.inl>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.inl>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.inl>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.inl>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

template class SOFA_GPU_CUDA_API component::mass::DiagonalMass<CudaVec3fTypes, float>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API component::mass::DiagonalMass<CudaVec3dTypes, double>;
#endif

// Register in the Factory
int DiagonalMassCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
    .add< component::mass::DiagonalMass<CudaVec3fTypes, float> >()

#ifdef SOFA_GPU_CUDA_DOUBLE
    .add< component::mass::DiagonalMass<CudaVec3dTypes,double> >()
// .add< component::mass::DiagonalMass<CudaVec3d1Types,double> >()
// .add< component::mass::DiagonalMass<CudaRigid3Types,sofa::defaulttype::Rigid3Mass> >()
 // SOFA_GPU_CUDA_DOUBLE
#endif
        ;


} // namespace mass

} // namespace component

} // namespace sofa

#endif // SOFA_GPU_CUDA_CUDADIAGONALMASS_CPP
