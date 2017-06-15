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
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETGEOMETRYALGORITHMS_CPP
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(TriangleSetGeometryAlgorithms)
int TriangleSetGeometryAlgorithmsClass = core::RegisterObject("Triangle set geometry algorithms")
#ifdef SOFA_FLOAT
        .add< TriangleSetGeometryAlgorithms<Vec3fTypes> >(true) // default template
#else
        .add< TriangleSetGeometryAlgorithms<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< TriangleSetGeometryAlgorithms<Vec3fTypes> >() // default template
#endif
#endif
#ifndef SOFA_FLOAT
        .add< TriangleSetGeometryAlgorithms<Vec2dTypes> >()
        .add< TriangleSetGeometryAlgorithms<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TriangleSetGeometryAlgorithms<Vec2fTypes> >()
        .add< TriangleSetGeometryAlgorithms<Vec1fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<Vec3dTypes>;
template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<Vec2dTypes>;
template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<Vec1dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<Vec3fTypes>;
template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<Vec2fTypes>;
template class SOFA_BASE_TOPOLOGY_API TriangleSetGeometryAlgorithms<Vec1fTypes>;
#endif


void SOFA_BASE_TOPOLOGY_API snapping_test_triangle(double epsilon, double alpha0, double alpha1, double alpha2,
        bool& is_snap_0, bool& is_snap_1, bool& is_snap_2)
{
    is_snap_0=false;
    is_snap_1=false;
    is_snap_2=false;

    if(alpha0>=alpha1 && alpha0>=alpha2)
    {
        is_snap_0=(alpha1+alpha2<epsilon);
    }
    else
    {
        if(alpha1>=alpha0 && alpha1>=alpha2)
        {
            is_snap_1=(alpha0+alpha2<epsilon);
        }
        else // alpha2>=alpha0 && alpha2>=alpha1
        {
            is_snap_2=(alpha0+alpha1<epsilon);
        }
    }
}

void SOFA_BASE_TOPOLOGY_API snapping_test_edge(double epsilon,	double alpha0, double alpha1,
        bool& is_snap_0, bool& is_snap_1)
{
    is_snap_0=false;
    is_snap_1=false;

    if(alpha0>=alpha1)
    {
        is_snap_0=(alpha1<epsilon);
    }
    else // alpha1>=alpha0
    {
        is_snap_1=(alpha0<epsilon);
    }
}

} // namespace topology

} // namespace component

} // namespace sofa
