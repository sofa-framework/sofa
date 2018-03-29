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
#ifndef SOFA_GPU_CUDA_CUDAUNIFORMMASS_H
#define SOFA_GPU_CUDA_CUDAUNIFORMMASS_H

#ifndef SOFA_DOUBLE //cuda only operates with float

#include "CudaTypes.h"
#include <SofaBaseMechanics/UniformMass.h>

namespace sofa
{

namespace component
{

namespace mass
{

// -- Mass interface
// CudaVec3f
template <>
void UniformMass<gpu::cuda::CudaVec3fTypes, float>::addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

template <>
void UniformMass<gpu::cuda::CudaVec3fTypes, float>::accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3fTypes, float>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

// CudaRigid3f
template <>
SReal UniformMass<gpu::cuda::CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const;

template <>
SReal UniformMass<gpu::cuda::CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getElementMass(unsigned int ) const;

template <>
void UniformMass<gpu::cuda::CudaRigid3fTypes, sofa::defaulttype::Rigid3fMass>::draw(const core::visual::VisualParams* vparams);

// CudaVec3f1
template <>
void UniformMass<gpu::cuda::CudaVec3f1Types, float>::addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

template <>
void UniformMass<gpu::cuda::CudaVec3f1Types, float>::accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3f1Types, float>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

//CudaRigid3f
template<>
void UniformMass<gpu::cuda::CudaRigid3fTypes, sofa::defaulttype::Rigid3fMass>::addMDx(const core::MechanicalParams *mparams, DataVecDeriv &f, const DataVecDeriv &dx, SReal factor);

template<>
void UniformMass<gpu::cuda::CudaRigid3fTypes, sofa::defaulttype::Rigid3fMass>::accFromF(const core::MechanicalParams *mparams, DataVecDeriv &a, const DataVecDeriv &f);

template<>
void UniformMass<gpu::cuda::CudaRigid3fTypes, sofa::defaulttype::Rigid3fMass>::addForce(const core::MechanicalParams *mparams, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &v);

#ifdef SOFA_GPU_CUDA_DOUBLE

// -- Mass interface
// CudaVec3d
template <>
void UniformMass<gpu::cuda::CudaVec3dTypes, SReal>::addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

template <>
void UniformMass<gpu::cuda::CudaVec3dTypes, SReal>::accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3dTypes, SReal>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

// template <>
// bool UniformMass<gpu::cuda::CudaVec3dTypes, double>::addBBox(SReal* minBBox, SReal* maxBBox);

// CudaRigid3d
template <>
SReal UniformMass<gpu::cuda::CudaRigid3dTypes,sofa::defaulttype::Rigid3dMass>::getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const;

template <>
SReal UniformMass<gpu::cuda::CudaRigid3dTypes,sofa::defaulttype::Rigid3dMass>::getElementMass(unsigned int ) const;

template <>
void UniformMass<gpu::cuda::CudaRigid3dTypes, sofa::defaulttype::Rigid3dMass>::draw(const core::visual::VisualParams* vparams);

// CudaVec3d1
template <>
void UniformMass<gpu::cuda::CudaVec3d1Types, double>::addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

template <>
void UniformMass<gpu::cuda::CudaVec3d1Types, double>::accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3d1Types, double>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

// template <>
// bool UniformMass<gpu::cuda::CudaVec3d1Types, double>::addBBox(SReal* minBBox, SReal* maxBBox);

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace mass

} // namespace component

} // namespace sofa

#endif

#endif
