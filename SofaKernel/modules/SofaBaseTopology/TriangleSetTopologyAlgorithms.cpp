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
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_CPP
#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;

int TriangleSetTopologyAlgorithmsClass = core::RegisterObject("Triangle set topology algorithms")
        .add< TriangleSetTopologyAlgorithms<Vec3Types> >(true) // default template
        .add< TriangleSetTopologyAlgorithms<Vec2Types> >()
        .add< TriangleSetTopologyAlgorithms<Vec1Types> >()
        ;


// methods specilizations declaration
template<> SOFA_BASE_TOPOLOGY_API
int TriangleSetTopologyAlgorithms<defaulttype::Vec2Types>::SplitAlongPath(PointID pa, Coord& a, PointID pb, Coord& b,
    sofa::helper::vector< sofa::core::topology::TopologyObjectType>& topoPath_list,
    sofa::helper::vector<ElemID>& indices_list,
    sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& coords_list,
    sofa::helper::vector<EdgeID>& new_edges, double epsilonSnapPath, double epsilonSnapBorder);
template<> SOFA_BASE_TOPOLOGY_API
int TriangleSetTopologyAlgorithms<defaulttype::Vec1Types>::SplitAlongPath(PointID pa, Coord& a, PointID pb, Coord& b,
    sofa::helper::vector< sofa::core::topology::TopologyObjectType>& topoPath_list,
    sofa::helper::vector<ElemID>& indices_list,
    sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& coords_list,
    sofa::helper::vector<EdgeID>& new_edges, double epsilonSnapPath, double epsilonSnapBorder);




template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<Vec3Types>;
template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<Vec2Types>;
template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<Vec1Types>;





// methods specilizations definition

template<>
int TriangleSetTopologyAlgorithms<defaulttype::Vec2Types>::SplitAlongPath(PointID , Coord& , PointID , Coord& ,
                                                              sofa::helper::vector< sofa::core::topology::TopologyObjectType>& ,
                                                              sofa::helper::vector<ElemID>& ,
                                                              sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& ,
                                                              sofa::helper::vector<EdgeID>& , double  , double )
{
    msg_warning() << "TriangleSetTopologyAlgorithms<defaulttype::Vec2Types>::SplitAlongPath not implemented";
    return 0;
}

template<>
int TriangleSetTopologyAlgorithms<defaulttype::Vec1Types>::SplitAlongPath(PointID , Coord& , PointID , Coord& ,
                                                              sofa::helper::vector< sofa::core::topology::TopologyObjectType>& ,
                                                              sofa::helper::vector<ElemID>& ,
                                                              sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& ,
                                                              sofa::helper::vector<EdgeID>& , double  , double )
{
    msg_warning() << "TriangleSetTopologyAlgorithms<defaulttype::Vec1Types>::SplitAlongPath not implemented";
    return 0;
}




} // namespace topology

} // namespace component

} // namespace sofa
