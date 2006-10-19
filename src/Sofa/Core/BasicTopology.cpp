#include "BasicTopology.h"

namespace Sofa
{

namespace Core
{

/** Question : shouldn't this be virtual, given this class has some virtual members?
 */
BasicTopology::~BasicTopology()
{
    if (topologyContainerObject)
        delete topologyContainerObject;
    if (topologyModifierObject)
        delete topologyModifierObject;
    if (topologyAlgorithmsObject)
        delete topologyAlgorithmsObject;
    if (geometryAlgorithmsObject)
        delete geometryAlgorithmsObject;
}

} // namespace Core

} // namespace Sofa


