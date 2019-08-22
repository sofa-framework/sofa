#ifndef SOFA_COMPONENT_TOPOLOGY_PARAMETRICTRIANGLETOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_PARAMETRICTRIANGLETOPOLOGYCONTAINER_H

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>

namespace sofa
{
namespace component
{
namespace topology
{

class ParametricTriangleTopologyContainer : public TriangleSetTopologyContainer
{
public:
    SOFA_CLASS(ParametricTriangleTopologyContainer,TriangleSetTopologyContainer);
    typedef defaulttype::Vector2 UV;
    typedef helper::vector<UV> SeqUV;

    void init();
    void reinit();

public:
    Data<SeqUV> d_uv; ///< The uv coordinates for every triangle vertices.

protected:
    ParametricTriangleTopologyContainer();
};


}

}

}


#endif // SOFA_COMPONENT_TOPOLOGY_PARAMETRICTRIANGLETOPOLOGYCONTAINER_H
