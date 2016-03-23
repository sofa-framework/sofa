/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MASS_CUDAMESHMATRIXMASS_H
#define SOFA_COMPONENT_MASS_CUDAMESHMATRIXMASS_H

#include "CudaTypes.h"
#include <SofaMiscForceField/MeshMatrixMass.h>

namespace sofa
{
namespace component
{
namespace mass
{

using namespace sofa::gpu::cuda;
using namespace sofa::component::mass;

template<>
class MeshMatrixMassInternalData<CudaVec2fTypes,float>
{
public:
    /// Cuda vector copying the vertex mass (enabling deviceRead)
    CudaVector<float> vMass;
};

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes, float>::copyVertexMass();

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes, float>::addMDx(const core::MechanicalParams*, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes, float>::addForce(const core::MechanicalParams*, DataVecDeriv& /*vf*/, const DataVecCoord& /* */, const DataVecDeriv& /* */);

template<>
void MeshMatrixMass<sofa::gpu::cuda::CudaVec2fTypes, float>::accFromF(const core::MechanicalParams*, DataVecDeriv& a, const DataVecDeriv& f);

} // namespace mass

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MASS_CUDAMESHMATRIXMASS_H
