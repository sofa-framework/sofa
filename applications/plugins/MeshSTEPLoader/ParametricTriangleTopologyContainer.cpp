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
    :TriangleSetTopologyContainer()
    ,d_uv(initData(&d_uv,"uv","The uv coordinates for every triangle vertices."))
{
}


void ParametricTriangleTopologyContainer::init()
{
    TriangleSetTopologyContainer::init();
}

void ParametricTriangleTopologyContainer::reinit()
{
    TriangleSetTopologyContainer::reinit();
}

}

}

}

