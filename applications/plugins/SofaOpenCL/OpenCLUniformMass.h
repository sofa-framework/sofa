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
#ifndef SOFAOPENCL_OPENCLUNIFORMMASS_H
#define SOFAOPENCL_OPENCLUNIFORMMASS_H

#include "OpenCLTypes.h"
#include <SofaBaseMechanics/UniformMass.h>

namespace sofa
{

namespace component
{

namespace mass
{


// -- Mass interface
template <>
void UniformMass<gpu::opencl::OpenCLVec3Types, SReal>::addMDx(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

template <>
void UniformMass<gpu::opencl::OpenCLVec3Types, SReal>::accFromF(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f);

template <>
void UniformMass<gpu::opencl::OpenCLVec3Types, SReal>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

//template <>
//bool UniformMass<gpu::opencl::OpenCLVec3Types, SReal>::addBBox(SReal* minBBox, SReal* maxBBox);

template <>
SReal UniformMass<gpu::opencl::OpenCLRigid3Types,sofa::defaulttype::Rigid3Mass>::getPotentialEnergy(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const;

template <>
SReal UniformMass<gpu::opencl::OpenCLRigid3Types,sofa::defaulttype::Rigid3Mass>::getElementMass(unsigned int ) const;

template <>
void UniformMass<gpu::opencl::OpenCLRigid3Types, sofa::defaulttype::Rigid3Mass>::draw(const sofa::core::visual::VisualParams* vparams);

template <>
void UniformMass<gpu::opencl::OpenCLVec3d1Types, SReal>::addMDx(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

template <>
void UniformMass<gpu::opencl::OpenCLVec3d1Types, SReal>::accFromF(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& a, const DataVecDeriv& f);

template <>
void UniformMass<gpu::opencl::OpenCLVec3d1Types, SReal>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

//template <>
//bool UniformMass<gpu::opencl::OpenCLVec3d1Types, SReal>::addBBox(SReal* minBBox, SReal* maxBBox);




} // namespace mass

} // namespace component

} // namespace sofa

#endif

