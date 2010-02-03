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
#ifndef SOFA_COMPONENT_COLLISION_CUDADIAGONALMASS_H
#define SOFA_COMPONENT_COLLISION_CUDADIAGONALMASS_H

#include "CudaTypes.h"
#include "CudaTypesBase.h"
#include <sofa/component/mass/DiagonalMass.h>

namespace sofa
{
namespace component
{
namespace mass
{

using namespace sofa::gpu::cuda;
using namespace sofa::component::linearsolver;

// template<>
// class DiagonalMassInternalData<CudaVec3Types,float> {
// public :
// 	typedef sofa::component::topology::PointData<float, CudaVector<float> > VecMass;
// 	typedef CudaVector<float> MassVector;
// };
//
// #ifdef SOFA_GPU_CUDA_DOUBLE
// template<>
// class DiagonalMassInternalData<CudaVec3dTypes,double> {
// public :
// 	typedef sofa::component::topology::PointData<double, CudaVector<double> > VecMass;
// 	typedef CudaVector<double> MassVector;
// };
// #endif


template <>
void DiagonalMass<gpu::cuda::CudaVec3fTypes, float>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor);

template <>
void DiagonalMass<gpu::cuda::CudaVec3fTypes, float>::accFromF(VecDeriv& a, const VecDeriv& f);

// template<>
// void DiagonalMass<gpu::cuda::CudaVec3fTypes, float>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&);

template<>
bool DiagonalMass<gpu::cuda::CudaVec3fTypes, float>::addBBox(double* minBBox, double* maxBBox);


#ifdef SOFA_GPU_CUDA_DOUBLE

// -- Mass interface
template <>
void DiagonalMass<gpu::cuda::CudaVec3dTypes, double>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor);

template <>
void DiagonalMass<gpu::cuda::CudaVec3dTypes, double>::accFromF(VecDeriv& a, const VecDeriv& f);

// template<>
// void DiagonalMass<gpu::cuda::CudaVec3dTypes, double>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&);

template<>
bool DiagonalMass<gpu::cuda::CudaVec3dTypes, double>::addBBox(double* minBBox, double* maxBBox);

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace mass

} // namespace component

} // namespace sofa

#endif
