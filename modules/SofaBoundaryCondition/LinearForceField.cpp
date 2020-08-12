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
#define SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_CPP

#include <sofa/core/ObjectFactory.h>
#include "LinearForceField.inl"

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

int LinearForceFieldClass = core::RegisterObject("Linearly interpolated force applied to given degrees of freedom")
        .add< LinearForceField<Vec3Types> >()
        .add< LinearForceField<Vec2Types> >()
        .add< LinearForceField<Vec1Types> >()
        .add< LinearForceField<Vec6Types> >()
        .add< LinearForceField<Rigid3Types> >()
// .add< LinearForceField<Rigid2Types> >()

        ;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec3Types>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec2Types>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec1Types>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec6Types>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Rigid3Types>;
// template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Rigid2Types>;


template <>
SReal LinearForceField<Rigid3Types>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const
{
    serr<<"LinearForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}
template <>
SReal LinearForceField<Rigid2Types>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const
{
    serr<<"LinearForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}


} // namespace forcefield

} // namespace component

} // namespace sofa
