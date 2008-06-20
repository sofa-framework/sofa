#include <sofa/core/componentmodel/topology/BaseTopology.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace topology
{

/** Question : shouldn't this be virtual, given this class has some virtual members?
         */
BaseTopology::~BaseTopology()
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


std::list<const TopologyChange *>::const_iterator BaseTopology::lastChange() const
{
    return m_topologyContainer->getChangeList().end();
}

std::list<const TopologyChange *>::const_iterator BaseTopology::firstChange() const
{
    return m_topologyContainer->getChangeList().begin();
}

std::list<const TopologyChange *>::const_iterator BaseTopology::lastStateChange() const
{
    return m_topologyContainer->getStateChangeList().end();
}

std::list<const TopologyChange *>::const_iterator BaseTopology::firstStateChange() const
{
    return m_topologyContainer->getStateChangeList().begin();
}

void BaseTopology::resetTopologyChangeList() const
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

void BaseTopology::resetStateChangeList() const
{
    getTopologyContainer()->resetStateChangeList();
}

void TopologyContainer::resetStateChangeList()
{
    std::list<const TopologyChange *>::iterator it=m_StateChangeList.begin();
    for (; it!=m_StateChangeList.end(); ++it)
    {
        delete (*it);
    }
    m_StateChangeList.erase(m_StateChangeList.begin(),m_StateChangeList.end());
}

} // namespace topology

} // namespace componentmodel

} // namespace core

} // namespace sofa

