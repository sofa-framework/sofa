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
#define SOFA_COMPONENT_FORCEFIELD_DIAGONALVELOCITYDAMPINGFORCEFIELD_CPP

#include "DiagonalVelocityDampingForceField.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

    namespace component
    {

        namespace forcefield
        {

            using namespace sofa::defaulttype;

            SOFA_DECL_CLASS(DiagonalVelocityDampingForceField)

                int DiagonalVelocityDampingForceFieldClass = core::RegisterObject("Diagonal velocity damping")
#ifndef SOFA_FLOAT
                .add< DiagonalVelocityDampingForceField<Vec3dTypes> >()
                .add< DiagonalVelocityDampingForceField<Vec2dTypes> >()
                .add< DiagonalVelocityDampingForceField<Vec1dTypes> >()
                .add< DiagonalVelocityDampingForceField<Vec6dTypes> >()
                .add< DiagonalVelocityDampingForceField<Rigid3dTypes> >()
                .add< DiagonalVelocityDampingForceField<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
                .add< DiagonalVelocityDampingForceField<Vec3fTypes> >()
                .add< DiagonalVelocityDampingForceField<Vec2fTypes> >()
                .add< DiagonalVelocityDampingForceField<Vec1fTypes> >()
                .add< DiagonalVelocityDampingForceField<Vec6fTypes> >()
                .add< DiagonalVelocityDampingForceField<Rigid3fTypes> >()
                .add< DiagonalVelocityDampingForceField<Rigid2fTypes> >()
                
#endif
                ;

#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec2dTypes>;
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec1dTypes>;
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec6dTypes>;
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Rigid3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec2fTypes>;
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec1fTypes>;
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Vec6fTypes>;
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Rigid3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API DiagonalVelocityDampingForceField<Rigid2fTypes>;
#endif

        } // namespace forcefield

    } // namespace component

} // namespace sofa
