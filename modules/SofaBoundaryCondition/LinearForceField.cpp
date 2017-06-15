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

SOFA_DECL_CLASS(LinearForceField)

int LinearForceFieldClass = core::RegisterObject("Linearly interpolated force applied to given degrees of freedom")
#ifndef SOFA_FLOAT
        .add< LinearForceField<Vec3dTypes> >()
        .add< LinearForceField<Vec2dTypes> >()
        .add< LinearForceField<Vec1dTypes> >()
        .add< LinearForceField<Vec6dTypes> >()
        .add< LinearForceField<Rigid3dTypes> >()
// .add< LinearForceField<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< LinearForceField<Vec3fTypes> >()
        .add< LinearForceField<Vec2fTypes> >()
        .add< LinearForceField<Vec1fTypes> >()
        .add< LinearForceField<Vec6fTypes> >()
        .add< LinearForceField<Rigid3fTypes> >()
// .add< LinearForceField<Rigid2fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec2dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec1dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec6dTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Rigid3dTypes>;
// template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec2fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec1fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Vec6fTypes>;
template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Rigid3fTypes>;
// template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Rigid2fTypes>;
#endif

#ifndef SOFA_FLOAT
template <>
SReal LinearForceField<Rigid3dTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const
{
    serr<<"LinearForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}
template <>
SReal LinearForceField<Rigid2dTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const
{
    serr<<"LinearForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}
#endif

#ifndef SOFA_DOUBLE
template <>
SReal LinearForceField<Rigid3fTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const
{
    serr<<"LinearForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}

template <>
SReal LinearForceField<Rigid2fTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const
{
    serr<<"LinearForceField::getPotentialEnergy-not-implemented !!!"<<sendl;
    return 0;
}
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
