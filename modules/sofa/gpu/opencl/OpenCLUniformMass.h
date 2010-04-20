/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_GPU_OPENCL_OPENCLUNIFORMMASS_H
#define SOFA_GPU_OPENCL_OPENCLUNIFORMMASS_H

#include "OpenCLTypes.h"
#include <sofa/component/mass/UniformMass.h>

namespace sofa
{

namespace component
{

namespace mass
{

// -- Mass interface
template <>
void UniformMass<gpu::opencl::OpenCLVec3fTypes, float>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor);

template <>
void UniformMass<gpu::opencl::OpenCLVec3fTypes, float>::accFromF(VecDeriv& a, const VecDeriv& f);

template <>
void UniformMass<gpu::opencl::OpenCLVec3fTypes, float>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&);

template <>
bool UniformMass<gpu::opencl::OpenCLVec3fTypes, float>::addBBox(double* minBBox, double* maxBBox);

template <>
double UniformMass<gpu::opencl::OpenCLRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getPotentialEnergy( const VecCoord& x ) const;

template <>
double UniformMass<gpu::opencl::OpenCLRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getElementMass(unsigned int ) const;

template <>
void UniformMass<gpu::opencl::OpenCLRigid3fTypes, Rigid3fMass>::draw();

template <>
void UniformMass<gpu::opencl::OpenCLVec3f1Types, float>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor);

template <>
void UniformMass<gpu::opencl::OpenCLVec3f1Types, float>::accFromF(VecDeriv& a, const VecDeriv& f);

template <>
void UniformMass<gpu::opencl::OpenCLVec3f1Types, float>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&);

template <>
bool UniformMass<gpu::opencl::OpenCLVec3f1Types, float>::addBBox(double* minBBox, double* maxBBox);


// -- Mass interface
template <>
void UniformMass<gpu::opencl::OpenCLVec3dTypes, double>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor);

template <>
void UniformMass<gpu::opencl::OpenCLVec3dTypes, double>::accFromF(VecDeriv& a, const VecDeriv& f);

template <>
void UniformMass<gpu::opencl::OpenCLVec3dTypes, double>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&);

template <>
bool UniformMass<gpu::opencl::OpenCLVec3dTypes, double>::addBBox(double* minBBox, double* maxBBox);

template <>
double UniformMass<gpu::opencl::OpenCLRigid3dTypes,sofa::defaulttype::Rigid3dMass>::getPotentialEnergy( const VecCoord& x ) const;

template <>
double UniformMass<gpu::opencl::OpenCLRigid3dTypes,sofa::defaulttype::Rigid3dMass>::getElementMass(unsigned int ) const;

template <>
void UniformMass<gpu::opencl::OpenCLRigid3dTypes, Rigid3dMass>::draw();

template <>
void UniformMass<gpu::opencl::OpenCLVec3d1Types, double>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor);

template <>
void UniformMass<gpu::opencl::OpenCLVec3d1Types, double>::accFromF(VecDeriv& a, const VecDeriv& f);

template <>
void UniformMass<gpu::opencl::OpenCLVec3d1Types, double>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&);

template <>
bool UniformMass<gpu::opencl::OpenCLVec3d1Types, double>::addBBox(double* minBBox, double* maxBBox);




} // namespace mass

} // namespace component

} // namespace sofa

#endif

