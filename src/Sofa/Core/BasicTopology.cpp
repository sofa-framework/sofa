#include "BasicTopology.h"

namespace Sofa
{

namespace Core
{

/** Question : shouldn't this be virtual, given this class has some virtual members?
 */
BasicTopology::~BasicTopology()
{
    if (m_topologyContainer)
        delete m_topologyContainer;
    if (m_topologyModifier)
        delete m_topologyModifier;
    if (m_topologyAlgorithms)
        delete m_topologyAlgorithms;
    if (m_geometryAlgorithms)
        delete m_geometryAlgorithms;
}



void BasicTopology::addTopologyChange(const TopologyChange &topologyChange)
{
    m_topologyContainer->getChangeList().push_back(topologyChange);
}



std::list<TopologyChange>::const_iterator BasicTopology::lastChange() const
{
    return m_topologyContainer->getChangeList().end();
}



std::list<TopologyChange>::const_iterator BasicTopology::firstChange() const
{
    return m_topologyContainer->getChangeList().begin();
}


} // namespace Core

} // namespace Sofa


