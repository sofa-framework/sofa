#ifndef SOFACOMBINATORIALMAPS_BASETOPOLOGY_CMTOPOLOGYENGINE_H_
#define SOFACOMBINATORIALMAPS_BASETOPOLOGY_CMTOPOLOGYENGINE_H_

#include <SofaCombinatorialMaps/config.h>

#include <SofaCombinatorialMaps/Core/CMBaseTopologyEngine.h>
#include <SofaCombinatorialMaps/Core/CMTopologyHandler.h>
#include <SofaCombinatorialMaps/Core/CMBaseTopologyData.h>

#include <SofaCombinatorialMaps/Core/CMapTopology.h>

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
class SOFA_COMBINATORIALMAPS_API TopologyEngineImpl : public sofa::core::cm_topology::TopologyEngine
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
			sofa::core::topology::CMapTopology* _topology,
            sofa::core::cm_topology::TopologyHandler* _topoHandler);

public:

    void init();

    void reinit();

	void doUpdate();

    void ApplyTopologyChanges();

	void registerTopology(sofa::core::topology::CMapTopology* _topology);

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
	sofa::core::topology::CMapTopology* m_topology;
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

#endif // SOFACOMBINATORIALMAPS_BASETOPOLOGY_CMTOPOLOGYENGINE_H_
