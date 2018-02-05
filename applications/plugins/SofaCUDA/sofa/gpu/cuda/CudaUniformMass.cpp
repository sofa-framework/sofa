/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include "CudaUniformMass.inl"
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaUniformMass)

int UniformMassCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::mass::UniformMass<CudaVec3fTypes,float> >()
        .add< component::mass::UniformMass<CudaVec3f1Types,float> >()
        .add< component::mass::UniformMass<CudaRigid3fTypes,sofa::defaulttype::RigidMass<3,float> > >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::mass::UniformMass<CudaVec3dTypes,double> >()
        .add< component::mass::UniformMass<CudaVec3d1Types,double> >()
        .add< component::mass::UniformMass<CudaRigid3dTypes,sofa::defaulttype::RigidMass<3,double> > >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
