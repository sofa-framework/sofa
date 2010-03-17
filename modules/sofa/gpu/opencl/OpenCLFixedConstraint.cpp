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
#include "OpenCLFixedConstraint.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace gpu
{

namespace opencl
{


SOFA_DECL_CLASS(OpenCLFixedConstraint)

int FixedConstraintOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< component::constraint::FixedConstraint<OpenCLVec3fTypes> >()
        .add< component::constraint::FixedConstraint<OpenCLVec3f1Types> >()
#ifdef SOFA_DEV
        .add< component::constraint::FixedConstraint<OpenCLRigid3fTypes> >()
#endif // SOFA_DEV
        .add< component::constraint::FixedConstraint<OpenCLVec3dTypes> >()
        .add< component::constraint::FixedConstraint<OpenCLVec3d1Types> >()
#ifdef SOFA_DEV
        .add< component::constraint::FixedConstraint<OpenCLRigid3dTypes> >()
#endif // SOFA_DEV
        ;

} // namespace opencl

} // namespace gpu

} // namespace sofa
