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
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PARABOLICCONSTRAINT_CPP
#include <SofaBoundaryCondition/ParabolicConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
//#include <sofa/helper/gl/Axis.h>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

SOFA_DECL_CLASS(ParabolicConstraint)


int ParabolicConstraintClass = core::RegisterObject("Apply a parabolic trajectory to given points")
#ifndef SOFA_FLOAT
        .add< ParabolicConstraint<defaulttype::Vec3dTypes> >()
        .add< ParabolicConstraint<defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ParabolicConstraint<defaulttype::Vec3fTypes> >()
        .add< ParabolicConstraint<defaulttype::Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class ParabolicConstraint<defaulttype::Rigid3dTypes>;
template class ParabolicConstraint<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class ParabolicConstraint<defaulttype::Rigid3fTypes>;
template class ParabolicConstraint<defaulttype::Vec3fTypes>;
#endif



} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

