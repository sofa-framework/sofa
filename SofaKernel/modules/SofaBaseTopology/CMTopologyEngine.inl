#ifndef CMTOPOLOGYENGINE_INL
#define CMTOPOLOGYENGINE_INL

#include <SofaBaseTopology/CMTopologyEngine.h>

//#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
//#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>


namespace sofa
{
namespace component
{
namespace cm_topology
{

template <typename VecT>
TopologyEngineImpl< VecT>::TopologyEngineImpl(t_topologicalData *_topologicalData,
        core::topology::MapTopology* _topology,
        sofa::core::cm_topology::TopologyHandler *_topoHandler) :
    m_topologicalData(_topologicalData),
    m_topology(NULL),
    m_topoHandler(_topoHandler),
    m_pointsLinked(false), m_edgesLinked(false), m_trianglesLinked(false),
    m_quadsLinked(false), m_tetrahedraLinked(false), m_hexahedraLinked(false)
{
    m_topology = _topology;

    if (m_topology == NULL)
        serr <<"Error: Topology is not dynamic" << sendl;

    if (m_topoHandler == NULL)
        serr <<"Error: Topology Handler not available" << sendl;
}

template <typename VecT>
void TopologyEngineImpl< VecT>::init()
{
    // A pointData is by default child of positionSet Data
    //this->linkToPointDataArray();  // already done while creating engine

    // Name creation
    if (m_prefix.empty()) m_prefix = "TopologyEngine_";
    m_data_name = this->m_topologicalData->getName();
    this->addOutput(this->m_topologicalData);

    sofa::core::cm_topology::TopologyEngine::init();

    // Register Engine in containter list
    //if (m_topology)
    //   m_topology->addTopologyEngine(this);
    //this->registerTopology(m_topology);
}


template <typename VecT>
void TopologyEngineImpl< VecT>::reinit()
{
    this->update();
}


template <typename VecT>
void TopologyEngineImpl< VecT>::update()
{
#ifndef NDEBUG // too much warnings
    sout << "TopologyEngine::update" << sendl;
    sout<< "Number of topological changes: " << m_changeList.getValue().size() << sendl;
#endif
    this->cleanDirty();
    this->ApplyTopologyChanges();
}


template <typename VecT>
void TopologyEngineImpl< VecT>::registerTopology(core::topology::MapTopology* _topology)
{
    m_topology =  _topology;

    if (m_topology == NULL)
    {
#ifndef NDEBUG // too much warnings
        serr <<"Error: Topology is not dynamic" << sendl;
#endif
        return;
    }
    else
        m_topology->addTopologyEngine(this);
}


template <typename VecT>
void TopologyEngineImpl< VecT>::registerTopology()
{
    if (m_topology == NULL)
    {
#ifndef NDEBUG // too much warnings
        serr <<"Error: Topology is not dynamic" << sendl;
#endif
        return;
    }
    else
        m_topology->addTopologyEngine(this);
}


template <typename VecT>
void TopologyEngineImpl< VecT>::ApplyTopologyChanges()
{
    // Rentre ici la premiere fois aussi....
    if(m_topoHandler)
    {
        m_topoHandler->ApplyTopologyChanges(m_changeList.getValue(), m_topology->getNbPoints());

        m_changeList.endEdit();
    }
}


/// Function to link DataEngine with Data array from topology
template <typename VecT>
void TopologyEngineImpl< VecT>::linkToPointDataArray()
{
	if (m_pointsLinked) // avoid second registration
		return;

	if (m_topology->getPointDataArray().getValue().size() > 0)
	{
		m_topology->getPointDataArray().addOutput(this);
		m_pointsLinked = true;
	}
}


template <typename VecT>
void TopologyEngineImpl< VecT>::linkToEdgeDataArray()
{
    if (m_edgesLinked) // avoid second registration
        return;

	if (m_topology->getTetrahedronDataArray().getValue().size() > 0)
	{
		m_topology->getEdgeDataArray().addOutput(this);
		m_edgesLinked = true;
	}
}


template <typename VecT>
void TopologyEngineImpl< VecT>::linkToTriangleDataArray()
{
	if (m_trianglesLinked) // avoid second registration
		return;

	if (m_topology->getTetrahedronDataArray().getValue().size() > 0)
	{
		m_topology->getTriangleDataArray().addOutput(this);
		m_trianglesLinked = true;
	}
}


template <typename VecT>
void TopologyEngineImpl< VecT>::linkToQuadDataArray()
{
    if (m_quadsLinked) // avoid second registration
        return;

	if (m_topology->getTetrahedronDataArray().getValue().size() > 0)
	{
		m_topology->getQuadDataArray().addOutput(this);
		m_quadsLinked = true;
	}
}


template <typename VecT>
void TopologyEngineImpl< VecT>::linkToTetrahedronDataArray()
{
    if (m_tetrahedraLinked) // avoid second registration
        return;

	if (m_topology->getTetrahedronDataArray().getValue().size() > 0)
	{
		m_topology->getTetrahedronDataArray().addOutput(this);
		m_tetrahedraLinked = true;
	}
}


template <typename VecT>
void TopologyEngineImpl< VecT>::linkToHexahedronDataArray()
{
    if (m_hexahedraLinked) // avoid second registration
        return;

	if (m_topology->getTetrahedronDataArray().getValue().size() > 0)
	{
		m_topology->getHexahedronDataArray().addOutput(this);
		m_hexahedraLinked = true;
	}
}





}// namespace cm_topology

} // namespace component

} // namespace sofa

#endif // CMTOPOLOGYENGINE_INL
