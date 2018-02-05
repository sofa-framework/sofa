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
#define SOFA_COMPONENT_MISC_EVALPOINTSDISTANCE_CPP
#include "EvalPointsDistance.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(EvalPointsDistance)

using namespace defaulttype;


int EvalPointsDistanceClass = core::RegisterObject("Periodically compute the distance between 2 set of points")
#ifndef SOFA_FLOAT
        .add< EvalPointsDistance<Vec3dTypes> >()
        .add< EvalPointsDistance<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< EvalPointsDistance<Vec3fTypes> >()
        .add< EvalPointsDistance<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_VALIDATION_API EvalPointsDistance<Vec3dTypes>;
template class SOFA_VALIDATION_API EvalPointsDistance<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_VALIDATION_API EvalPointsDistance<Vec3fTypes>;
template class SOFA_VALIDATION_API EvalPointsDistance<Rigid3fTypes>;
#endif

} // namespace misc

} // namespace component

} // namespace sofa
