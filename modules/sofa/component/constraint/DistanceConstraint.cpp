/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/constraint/DistanceConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

///TODO: handle combinaison of Rigid and Deformable bodies.

SOFA_DECL_CLASS(DistanceConstraint)

int DistanceConstraintClass = core::RegisterObject("Maintain constant the length of some edges of a pair of objects")
#ifndef SOFA_FLOAT
        .add< DistanceConstraint<Vec3dTypes> >()
        .add< DistanceConstraint<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< DistanceConstraint<Vec3fTypes> >()
        .add< DistanceConstraint<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class DistanceConstraint<Vec3dTypes>;
template class DistanceConstraint<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class DistanceConstraint<Vec3fTypes>;
template class DistanceConstraint<Rigid3fTypes>;
#endif



#ifndef SOFA_FLOAT
template<>
Rigid3dTypes::Deriv DistanceConstraint<Rigid3dTypes>::getDirection(const Edge &e, const VecCoord &x1, const VecCoord &x2)
{
    Vector3 V12=(x2[e[1]].getCenter() - x1[e[0]].getCenter()); V12.normalize();
    return Deriv(V12, Vector3());
}
#endif

#ifndef SOFA_DOUBLE
template<>
Rigid3fTypes::Deriv DistanceConstraint<Rigid3fTypes>::getDirection(const Edge &e, const VecCoord &x1, const VecCoord &x2)
{
    Vector3 V12=(x2[e[1]].getCenter() - x1[e[0]].getCenter()); V12.normalize();
    return Deriv(V12, Vector3());
}
#endif
} // namespace constraint

} // namespace component

} // namespace sofa

