#ifndef CMTOPOLOGYENGINE_H
#define CMTOPOLOGYENGINE_H

#include "config.h"

#include <sofa/core/topology/CMBaseTopologyEngine.h>
#include <sofa/core/topology/CMTopologyHandler.h>
#include <sofa/core/topology/CMBaseTopologyData.h>

#include <sofa/core/topology/MapTopology.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/list.h>


namespace sofa
{

namespace component
{

namespace cm_topology
{



////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class T>
class TopologyEngineImpl : public sofa::core::cm_topology::TopologyEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TopologyEngineImpl,T), sofa::core::cm_topology::TopologyEngine);
    typedef T container_type;
//    typedef typename container_type::value_type value_type;
    typedef sofa::core::cm_topology::BaseTopologyData<T> t_topologicalData;



//    typedef core::topology::BaseMeshTopology::Point Point;
//    typedef core::topology::BaseMeshTopology::Edge Edge;
//    typedef core::topology::BaseMeshTopology::Triangle Triangle;
//    typedef core::topology::BaseMeshTopology::Quad Quad;
//    typedef core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
//    typedef core::topology::BaseMeshTopology::Hexahedron Hexahedron;


protected:
    //TopologyEngineImpl();

    TopologyEngineImpl(t_topologicalData* _topologicalData,
            sofa::core::topology::MapTopology* _topology,
            sofa::core::cm_topology::TopologyHandler* _topoHandler);

public:

    void init();

    void reinit();

    void update();

    void ApplyTopologyChanges();

    void registerTopology(sofa::core::topology::MapTopology* _topology);

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
    sofa::core::topology::MapTopology* m_topology;
    sofa::core::cm_topology::TopologyHandler* m_topoHandler;

public:
    bool m_pointsLinked;
    bool m_edgesLinked;
    bool m_trianglesLinked;
    bool m_quadsLinked;
    bool m_tetrahedraLinked;
    bool m_hexahedraLinked;

};


} // namespace cm_topology

} // namespace component

} // namespace sofa

#endif // CMTOPOLOGYENGINE_H
