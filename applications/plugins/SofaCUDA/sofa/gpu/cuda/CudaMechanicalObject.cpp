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
#ifndef SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_CPP
#define SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_CPP


#include "CudaTypes.h"
#include "CudaMechanicalObject.inl"
#include <sofa/core/ObjectFactory.h>
#include <SofaBaseMechanics/MappedObject.inl>
#include <sofa/core/State.inl>

namespace sofa
{

namespace component
{

namespace container
{
// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.
template class MechanicalObject<CudaVec1fTypes>;
template class MechanicalObject<CudaVec2fTypes>;
template class MechanicalObject<CudaVec3fTypes>;
template class MechanicalObject<CudaVec3f1Types>;
template class MechanicalObject<CudaVec6fTypes>;
template class MechanicalObject<CudaRigid3fTypes>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template class MechanicalObject<CudaVec3dTypes>;
template class MechanicalObject<CudaVec3d1Types>;
template class MechanicalObject<CudaVec6dTypes>;
template class MechanicalObject<CudaRigid3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE

}

} // namespace component

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaMechanicalObject)

int MechanicalObjectCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::container::MechanicalObject<CudaVec1fTypes> >()
        .add< component::container::MechanicalObject<CudaVec2fTypes> >()
        .add< component::container::MechanicalObject<CudaVec3fTypes> >()
        .add< component::container::MechanicalObject<CudaVec3f1Types> >()
        .add< component::container::MechanicalObject<CudaVec6fTypes> >()
        .add< component::container::MechanicalObject<CudaRigid3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::container::MechanicalObject<CudaVec3dTypes> >()
        .add< component::container::MechanicalObject<CudaVec3d1Types> >()
        .add< component::container::MechanicalObject<CudaVec6dTypes> >()
        .add< component::container::MechanicalObject<CudaRigid3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

int MappedObjectCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::container::MappedObject<CudaVec1fTypes> >()
        .add< component::container::MappedObject<CudaVec2fTypes> >()
        .add< component::container::MappedObject<CudaVec3fTypes> >()
        .add< component::container::MappedObject<CudaVec3f1Types> >()
        .add< component::container::MappedObject<CudaVec6fTypes> >()
        .add< component::container::MappedObject<CudaRigid3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::container::MappedObject<CudaVec3dTypes> >()
        .add< component::container::MappedObject<CudaVec3d1Types> >()
        .add< component::container::MappedObject<CudaVec6dTypes> >()
        .add< component::container::MappedObject<CudaRigid3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa

#endif // SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_CPP
