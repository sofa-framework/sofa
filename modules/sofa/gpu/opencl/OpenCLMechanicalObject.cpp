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
#include "OpenCLTypes.h"
#include "OpenCLMechanicalObject.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/container/MappedObject.inl>

namespace sofa
{

namespace component
{

namespace container
{
// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.
template class MechanicalObject<gpu::opencl::OpenCLVec3fTypes>;
template class MechanicalObject<gpu::opencl::OpenCLVec3dTypes>;

}

} // namespace component

namespace gpu
{

namespace opencl
{

SOFA_DECL_CLASS(OpenCLMechanicalObject)

int MechanicalObjectOpenCLClass = core::RegisterObject("Supports GPU-side computations using OpenCL")
        .add< component::container::MechanicalObject<OpenCLVec3fTypes> >()
        .add< component::container::MechanicalObject<OpenCLVec3dTypes> >()
        ;

int MappedObjectOpenCLClass = core::RegisterObject("Supports GPU-side computations using OpenCL")
        .add< component::container::MappedObject<OpenCLVec3fTypes> >()
        .add< component::container::MappedObject<OpenCLVec3dTypes> >()
        ;

} // namespace opencl

} // namespace gpu

} // namespace sofa
