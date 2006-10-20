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

} // namespace Core

} // namespace Sofa


