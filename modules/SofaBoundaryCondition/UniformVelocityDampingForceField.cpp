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
#define SOFA_COMPONENT_FORCEFIELD_UNIFORMVELOCITYDAMPINGFORCEFIELD_CPP

#include "UniformVelocityDampingForceField.inl"
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

            SOFA_DECL_CLASS(UniformVelocityDampingForceField)

                int UniformVelocityDampingForceFieldClass = core::RegisterObject("Uniform velocity damping")
#ifndef SOFA_FLOAT
                .add< UniformVelocityDampingForceField<Vec3dTypes> >()
                .add< UniformVelocityDampingForceField<Vec2dTypes> >()
                .add< UniformVelocityDampingForceField<Vec1dTypes> >()
                .add< UniformVelocityDampingForceField<Vec6dTypes> >()
                .add< UniformVelocityDampingForceField<Rigid3dTypes> >()
                .add< UniformVelocityDampingForceField<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
                .add< UniformVelocityDampingForceField<Vec3fTypes> >()
                .add< UniformVelocityDampingForceField<Vec2fTypes> >()
                .add< UniformVelocityDampingForceField<Vec1fTypes> >()
                .add< UniformVelocityDampingForceField<Vec6fTypes> >()
                .add< UniformVelocityDampingForceField<Rigid3fTypes> >()
                .add< UniformVelocityDampingForceField<Rigid2fTypes> >()
#endif
                ;


#ifndef SOFA_FLOAT
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec3dTypes>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec2dTypes>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec1dTypes>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec6dTypes>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Rigid3dTypes>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec3fTypes>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec2fTypes>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec1fTypes>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec6fTypes>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Rigid3fTypes>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Rigid2fTypes>;
#endif

        } // namespace forcefield

    } // namespace component

} // namespace sofa
