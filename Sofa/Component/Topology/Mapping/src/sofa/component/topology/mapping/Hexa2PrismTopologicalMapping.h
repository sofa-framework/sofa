
#pragma once

#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/component/topology/mapping/config.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::topology::mapping
{

class Hexa2PrismTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(Hexa2PrismTopologicalMapping, sofa::core::topology::TopologicalMapping);

    using Index = sofa::core::topology::BaseMeshTopology::Index;
    using Prism = sofa::core::topology::BaseMeshTopology::Prism;

    Hexa2PrismTopologicalMapping();
    virtual ~Hexa2PrismTopologicalMapping();

    virtual void init() override;
    virtual Index getFromIndex(Index ind) override;
    virtual void updateTopologicalMappingTopDown() override;

private:
    sofa::Data<bool> d_swapping;
};

} // namespace sofa::component::topology::mapping
