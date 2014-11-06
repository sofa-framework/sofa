/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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

#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYENGINE_H
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYENGINE_H

#include <sofa/core/topology/BaseTopologyEngine.h>
#include <sofa/core/topology/TopologyHandler.h>
#include <sofa/core/topology/BaseTopologyData.h>

#include <sofa/core/topology/BaseTopology.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/list.h>

namespace sofa
{

namespace component
{

namespace topology
{

// Define topology elements
using core::topology::BaseMeshTopology;
typedef BaseMeshTopology::Point Point;
typedef BaseMeshTopology::Edge Edge;
typedef BaseMeshTopology::Triangle Triangle;
typedef BaseMeshTopology::Quad Quad;
typedef BaseMeshTopology::Tetrahedron Tetrahedron;
typedef BaseMeshTopology::Hexahedron Hexahedron;



////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT>
class TopologyEngineImpl : public sofa::core::topology::TopologyEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TopologyEngineImpl,VecT), sofa::core::topology::TopologyEngine);
    typedef VecT container_type;
    typedef typename container_type::value_type value_type;
    typedef sofa::core::topology::BaseTopologyData<VecT> t_topologicalData;

protected:
    //TopologyEngineImpl();

    TopologyEngineImpl(t_topologicalData* _topologicalData,
            sofa::core::topology::BaseMeshTopology* _topology,
            sofa::core::topology::TopologyHandler* _topoHandler);

public:

    void init();

    void reinit();

    void update();

    void ApplyTopologyChanges();

    void registerTopology(sofa::core::topology::BaseMeshTopology* _topology);

    void registerTopology();

    void registerTopologicalData(t_topologicalData *topologicalData) {m_topologicalData = topologicalData;}


    /// Function to link DataEngine with Data array from topology
    void linkToPointDataArray();
    void linkToEdgeDataArray();
    void linkToTriangleDataArray();
    void linkToQuadDataArray();
    void linkToTetrahedronDataArray();
    void linkToHexahedronDataArray();

protected:
    t_topologicalData* m_topologicalData;
    sofa::core::topology::TopologyContainer* m_topology;
    sofa::core::topology::TopologyHandler* m_topoHandler;

public:
    bool m_pointsLinked;
    bool m_edgesLinked;
    bool m_trianglesLinked;
    bool m_quadsLinked;
    bool m_tetrahedraLinked;
    bool m_hexahedraLinked;

};


} // namespace topology

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYENGINE_H
