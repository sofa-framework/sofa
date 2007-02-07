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


std::list<const TopologyChange *>::const_iterator BasicTopology::lastChange() const
{
    return m_topologyContainer->getChangeList().end();
}



std::list<const TopologyChange *>::const_iterator BasicTopology::firstChange() const
{
    return m_topologyContainer->getChangeList().begin();
}
void BasicTopology::resetTopologyChangeList() const
{
    getTopologyContainer()->resetTopologyChangeList();
}

void TopologyContainer::resetTopologyChangeList()
{
    std::list<const TopologyChange *>::iterator it=m_changeList.begin();
    for (; it!=m_changeList.end(); ++it)
    {
        delete (*it);
    }
    m_changeList.erase(m_changeList.begin(),m_changeList.end());
}

} // namespace Core

} // namespace Sofa


