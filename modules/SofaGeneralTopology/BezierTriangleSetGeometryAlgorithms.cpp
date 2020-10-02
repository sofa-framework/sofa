/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#define SOFA_COMPONENT_TOPOLOGY_BEZIERTRIANGLESETGEOMETRYALGORITHMS_CPP
#include <SofaGeneralTopology/BezierTriangleSetGeometryAlgorithms.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
SOFA_DECL_CLASS(BezierTriangleSetGeometryAlgorithms)
int BezierTriangleSetGeometryAlgorithmsClass = core::RegisterObject("Bezier Triangle set geometry algorithms")
#ifdef SOFA_FLOAT
        .add< BezierTriangleSetGeometryAlgorithms<Vec3fTypes> >(true) // default template
#else
        .add< BezierTriangleSetGeometryAlgorithms<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< BezierTriangleSetGeometryAlgorithms<Vec3fTypes> >() // default template
#endif
#endif
#ifndef SOFA_FLOAT
        .add< BezierTriangleSetGeometryAlgorithms<Vec2dTypes> >()
        .add< BezierTriangleSetGeometryAlgorithms<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< BezierTriangleSetGeometryAlgorithms<Vec2fTypes> >()
        .add< BezierTriangleSetGeometryAlgorithms<Vec1fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<Vec3dTypes>;
template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<Vec2dTypes>;
template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<Vec1dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<Vec3fTypes>;
template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<Vec2fTypes>;
template class SOFA_GENERAL_TOPOLOGY_API BezierTriangleSetGeometryAlgorithms<Vec1fTypes>;
#endif

} // namespace topology

} // namespace component

} // namespace sofa

