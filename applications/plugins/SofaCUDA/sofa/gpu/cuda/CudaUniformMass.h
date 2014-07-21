/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
void UniformMass<gpu::cuda::CudaVec3fTypes, float>::addMDx(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx, double factor);

template <>
void UniformMass<gpu::cuda::CudaVec3fTypes, float>::accFromF(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3fTypes, float>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

// CudaRigid3f
template <>
double UniformMass<gpu::cuda::CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getPotentialEnergy(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const;

template <>
double UniformMass<gpu::cuda::CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getElementMass(unsigned int ) const;

template <>
void UniformMass<gpu::cuda::CudaRigid3fTypes, sofa::defaulttype::Rigid3fMass>::draw(const core::visual::VisualParams* vparams);

// CudaVec3f1
template <>
void UniformMass<gpu::cuda::CudaVec3f1Types, float>::addMDx(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx, double factor);

template <>
void UniformMass<gpu::cuda::CudaVec3f1Types, float>::accFromF(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3f1Types, float>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

//CudaRigid3f
template<>
void UniformMass<gpu::cuda::CudaRigid3fTypes, sofa::defaulttype::Rigid3fMass>::addMDx(const core::MechanicalParams *mparams /* PARAMS FIRST */, DataVecDeriv &f, const DataVecDeriv &dx, double factor);

template<>
void UniformMass<gpu::cuda::CudaRigid3fTypes, sofa::defaulttype::Rigid3fMass>::accFromF(const core::MechanicalParams *mparams /* PARAMS FIRST */, DataVecDeriv &a, const DataVecDeriv &f);

template<>
void UniformMass<gpu::cuda::CudaRigid3fTypes, sofa::defaulttype::Rigid3fMass>::addForce(const core::MechanicalParams *mparams /* PARAMS FIRST */, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &v);

#ifdef SOFA_GPU_CUDA_DOUBLE

// -- Mass interface
// CudaVec3d
template <>
void UniformMass<gpu::cuda::CudaVec3dTypes, double>::addMDx(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx, double factor);

template <>
void UniformMass<gpu::cuda::CudaVec3dTypes, double>::accFromF(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3dTypes, double>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

// template <>
// bool UniformMass<gpu::cuda::CudaVec3dTypes, double>::addBBox(double* minBBox, double* maxBBox);

// CudaRigid3d
template <>
double UniformMass<gpu::cuda::CudaRigid3dTypes,sofa::defaulttype::Rigid3dMass>::getPotentialEnergy(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const;

template <>
double UniformMass<gpu::cuda::CudaRigid3dTypes,sofa::defaulttype::Rigid3dMass>::getElementMass(unsigned int ) const;

template <>
void UniformMass<gpu::cuda::CudaRigid3dTypes, Rigid3dMass>::draw(const core::visual::VisualParams* vparams);

// CudaVec3d1
template <>
void UniformMass<gpu::cuda::CudaVec3d1Types, double>::addMDx(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx, double factor);

template <>
void UniformMass<gpu::cuda::CudaVec3d1Types, double>::accFromF(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f);

template <>
void UniformMass<gpu::cuda::CudaVec3d1Types, double>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

// template <>
// bool UniformMass<gpu::cuda::CudaVec3d1Types, double>::addBBox(double* minBBox, double* maxBBox);

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace mass

} // namespace component

} // namespace sofa

#endif

#endif
