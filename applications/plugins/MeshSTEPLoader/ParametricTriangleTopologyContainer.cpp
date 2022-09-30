#include "ParametricTriangleTopologyContainer.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace topology
{

using namespace sofa::core;

int ParametricTriangleTopologyContainerClass = RegisterObject("Topology container for triangles \
                                                              with parametric coordinates")
        .add<ParametricTriangleTopologyContainer>();

ParametricTriangleTopologyContainer::ParametricTriangleTopologyContainer()
    : container::dynamic::TriangleSetTopologyContainer()
    ,d_uv(initData(&d_uv,"uv","The uv coordinates for every triangle vertices."))
{
}


void ParametricTriangleTopologyContainer::init()
{
    container::dynamic::TriangleSetTopologyContainer::init();
}

void ParametricTriangleTopologyContainer::reinit()
{
    container::dynamic::TriangleSetTopologyContainer::reinit();
}

}

}

}

