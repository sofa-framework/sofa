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
#include "CudaTypes.h"
#include "CudaDiagonalMass.inl"
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{


SOFA_DECL_CLASS(CudaDiagonalMass)
// Register in the Factory
int DiagonalMassCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
#ifndef SOFA_FLOAT
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::mass::DiagonalMass<CudaVec3dTypes,double> >()
// .add< component::mass::DiagonalMass<CudaVec3d1Types,double> >()
// .add< component::mass::DiagonalMass<CudaRigid3dTypes,sofa::defaulttype::Rigid3dMass> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif
#ifndef SOFA_DOUBLE
        .add< component::mass::DiagonalMass<CudaVec3fTypes,float> >()
// .add< component::mass::DiagonalMass<CudaVec3f1Types,float> >()
// .add< component::mass::DiagonalMass<CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass> >()
#endif
        ;


} // namespace mass

} // namespace component

} // namespace sofa

