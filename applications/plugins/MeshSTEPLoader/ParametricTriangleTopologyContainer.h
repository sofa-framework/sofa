#ifndef SOFA_COMPONENT_TOPOLOGY_PARAMETRICTRIANGLETOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_PARAMETRICTRIANGLETOPOLOGYCONTAINER_H

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/vector.h>

namespace sofa
{
namespace component
{
namespace topology
{

class ParametricTriangleTopologyContainer : public container::dynamic::TriangleSetTopologyContainer
{
public:
    SOFA_CLASS(ParametricTriangleTopologyContainer, container::dynamic::TriangleSetTopologyContainer);
    typedef type::Vec2 UV;
    typedef type::vector<UV> SeqUV;

    void init() override;
    void reinit() override;

public:
    Data<SeqUV> d_uv; ///< The uv coordinates for every triangle vertices.

protected:
    ParametricTriangleTopologyContainer();
};


}

}

}


#endif // SOFA_COMPONENT_TOPOLOGY_PARAMETRICTRIANGLETOPOLOGYCONTAINER_H
