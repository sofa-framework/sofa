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
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_CPP
#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.h>
#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(TriangleSetTopologyAlgorithms)
int TriangleSetTopologyAlgorithmsClass = core::RegisterObject("Triangle set topology algorithms")
#ifdef SOFA_FLOAT
        .add< TriangleSetTopologyAlgorithms<Vec3fTypes> >(true) // default template
#else
        .add< TriangleSetTopologyAlgorithms<Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< TriangleSetTopologyAlgorithms<Vec3fTypes> >() // default template
#endif
#endif
#ifndef SOFA_FLOAT
        .add< TriangleSetTopologyAlgorithms<Vec2dTypes> >()
        .add< TriangleSetTopologyAlgorithms<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TriangleSetTopologyAlgorithms<Vec2fTypes> >()
        .add< TriangleSetTopologyAlgorithms<Vec1fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<Vec3dTypes>;
template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<Vec2dTypes>;
template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<Vec1dTypes>;
#endif

#ifndef SOFA_DOUBLE
template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<Vec3fTypes>;
template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<Vec2fTypes>;
template class SOFA_BASE_TOPOLOGY_API TriangleSetTopologyAlgorithms<Vec1fTypes>;
#endif









#ifndef SOFA_FLOAT


template<> SOFA_BASE_TOPOLOGY_API
int TriangleSetTopologyAlgorithms<defaulttype::Vec2dTypes>::SplitAlongPath(unsigned int , Coord& , unsigned int , Coord& ,
                                                              sofa::helper::vector< sofa::core::topology::TopologyObjectType>& ,
                                                              sofa::helper::vector<unsigned int>& ,
                                                              sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& ,
                                                              sofa::helper::vector<EdgeID>& , double  , double )
{
    serr<<"TriangleSetTopologyAlgorithms<defaulttype::Vec2dTypes>::SplitAlongPath not implemented"<<sendl;
    return 0;
}
template<> SOFA_BASE_TOPOLOGY_API
int TriangleSetTopologyAlgorithms<defaulttype::Vec1dTypes>::SplitAlongPath(unsigned int , Coord& , unsigned int , Coord& ,
                                                              sofa::helper::vector< sofa::core::topology::TopologyObjectType>& ,
                                                              sofa::helper::vector<unsigned int>& ,
                                                              sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& ,
                                                              sofa::helper::vector<EdgeID>& , double  , double )
{
    serr<<"TriangleSetTopologyAlgorithms<defaulttype::Vec1dTypes>::SplitAlongPath not implemented"<<sendl;
    return 0;
}

#endif


#ifndef SOFA_DOUBLE

template<> SOFA_BASE_TOPOLOGY_API
int TriangleSetTopologyAlgorithms<defaulttype::Vec2fTypes>::SplitAlongPath(unsigned int , Coord& , unsigned int , Coord& ,
                                                              sofa::helper::vector< sofa::core::topology::TopologyObjectType>& ,
                                                              sofa::helper::vector<unsigned int>& ,
                                                              sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& ,
                                                              sofa::helper::vector<EdgeID>& , double  , double )
{
    sout<<"TriangleSetTopologyAlgorithms<defaulttype::Vec2fTypes>::SplitAlongPath not implemented"<<sendl;
    return 0;
}
template<> SOFA_BASE_TOPOLOGY_API
int TriangleSetTopologyAlgorithms<defaulttype::Vec1fTypes>::SplitAlongPath(unsigned int , Coord& , unsigned int , Coord& ,
                                                              sofa::helper::vector< sofa::core::topology::TopologyObjectType>& ,
                                                              sofa::helper::vector<unsigned int>& ,
                                                              sofa::helper::vector< sofa::defaulttype::Vec<3, double> >& ,
                                                              sofa::helper::vector<EdgeID>& , double  , double )
{
    sout<<"TriangleSetTopologyAlgorithms<defaulttype::Vec1fTypes>::SplitAlongPath not implemented"<<sendl;
    return 0;
}


#endif




} // namespace topology

} // namespace component

} // namespace sofa
