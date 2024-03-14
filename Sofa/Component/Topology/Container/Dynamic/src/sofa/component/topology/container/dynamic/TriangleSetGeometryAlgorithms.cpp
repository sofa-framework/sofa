/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.inl>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology::container::dynamic
{
using namespace sofa::defaulttype;

int TriangleSetGeometryAlgorithmsClass = core::RegisterObject("Triangle set geometry algorithms")
        .add< TriangleSetGeometryAlgorithms<Vec3Types> >(true) // default template
        .add< TriangleSetGeometryAlgorithms<Vec2Types> >()
        ;



// methods specilizations declaration
template<> SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API
int TriangleSetGeometryAlgorithms<defaulttype::Vec2Types>::SplitAlongPath(PointID pa, Coord& a, PointID pb, Coord& b,
    sofa::type::vector< sofa::geometry::ElementType>& topoPath_list,
    sofa::type::vector<ElemID>& indices_list,
    sofa::type::vector< sofa::type::Vec3 >& coords_list,
    sofa::type::vector<EdgeID>& new_edges, SReal epsilonSnapPath, SReal epsilonSnapBorder);
template<> SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API
int TriangleSetGeometryAlgorithms<defaulttype::Vec1Types>::SplitAlongPath(PointID pa, Coord& a, PointID pb, Coord& b,
    sofa::type::vector< sofa::geometry::ElementType>& topoPath_list,
    sofa::type::vector<ElemID>& indices_list,
    sofa::type::vector< sofa::type::Vec3 >& coords_list,
    sofa::type::vector<EdgeID>& new_edges, SReal epsilonSnapPath, SReal epsilonSnapBorder);



template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API TriangleSetGeometryAlgorithms<Vec3Types>;
template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API TriangleSetGeometryAlgorithms<Vec2Types>;


template<>
int TriangleSetGeometryAlgorithms<defaulttype::Vec2Types>::SplitAlongPath(PointID, Coord&, PointID, Coord&,
    sofa::type::vector< sofa::geometry::ElementType>&,
    sofa::type::vector<ElemID>&,
    sofa::type::vector< sofa::type::Vec3 >&,
    sofa::type::vector<EdgeID>&, SReal, SReal)
{
    msg_warning() << "TriangleSetTopologyAlgorithms<defaulttype::Vec2Types>::SplitAlongPath not implemented";
    return 0;
}

template<>
int TriangleSetGeometryAlgorithms<defaulttype::Vec1Types>::SplitAlongPath(PointID, Coord&, PointID, Coord&,
    sofa::type::vector< sofa::geometry::ElementType>&,
    sofa::type::vector<ElemID>&,
    sofa::type::vector< sofa::type::Vec3 >&,
    sofa::type::vector<EdgeID>&, SReal, SReal)
{
    msg_warning() << "TriangleSetTopologyAlgorithms<defaulttype::Vec1Types>::SplitAlongPath not implemented";
    return 0;
}



void SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API snapping_test_triangle(SReal epsilon, SReal alpha0, SReal alpha1, SReal alpha2,
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

void SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API snapping_test_edge(SReal epsilon,	SReal alpha0, SReal alpha1,
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

} //namespace sofa::component::topology::container::dynamic
