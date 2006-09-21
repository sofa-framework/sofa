#include "BasicTopology.h"

namespace Sofa
{

namespace Core
{
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


