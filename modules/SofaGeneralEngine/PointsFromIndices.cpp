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
#define SOFA_COMPONENT_ENGINE_POINTSFROMINDICES_CPP
#include <SofaGeneralEngine/PointsFromIndices.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(PointsFromIndices)

int PointsFromIndicesClass = core::RegisterObject("Find the points given a list of indices")
#ifndef SOFA_FLOAT
        .add< PointsFromIndices<Vec3dTypes> >()
// .add< PointsFromIndices<Rigid3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< PointsFromIndices<Vec3fTypes> >()
// .add< PointsFromIndices<Rigid3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_ENGINE_API PointsFromIndices<Vec3dTypes>;
// template class SOFA_GENERAL_ENGINE_API PointsFromIndices<Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_ENGINE_API PointsFromIndices<Vec3fTypes>;
// template class SOFA_GENERAL_ENGINE_API PointsFromIndices<Rigid3fTypes>;
#endif //SOFA_DOUBLE


} // namespace constraint

} // namespace component

} // namespace sofa

