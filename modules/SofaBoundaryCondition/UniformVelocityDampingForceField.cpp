/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

                int UniformVelocityDampingForceFieldClass = core::RegisterObject("Uniform velocity damping")
                .add< UniformVelocityDampingForceField<Vec3Types> >()
                .add< UniformVelocityDampingForceField<Vec2Types> >()
                .add< UniformVelocityDampingForceField<Vec1Types> >()
                .add< UniformVelocityDampingForceField<Vec6Types> >()
                .add< UniformVelocityDampingForceField<Rigid3Types> >()
                .add< UniformVelocityDampingForceField<Rigid2Types> >()

                ;


            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec3Types>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec2Types>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec1Types>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Vec6Types>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Rigid3Types>;
            template class SOFA_BOUNDARY_CONDITION_API UniformVelocityDampingForceField<Rigid2Types>;


        } // namespace forcefield

    } // namespace component

} // namespace sofa
