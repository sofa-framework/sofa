#ifndef SOFA_CORE_TOPOLOGICALMAPPING_H
#define SOFA_CORE_TOPOLOGICALMAPPING_H

#include "BasicMapping.h"
#include "BasicTopology.h"

namespace Sofa
{

namespace Core
{

class TopologicalMapping : public BasicMapping
{

protected:
    BasicTopology* mainTopology;
    BasicTopology* specificTopology;

public:
    TopologicalMapping(BasicTopology* from, BasicTopology* to);

    virtual ~TopologicalMapping();

    Abstract::BaseObject* getFrom()
    {
        return mainTopology;
    }
    Abstract::BaseObject* getTo()
    {
        return specificTopology;
    }

    virtual void updateSpecificTopology()=0;

    virtual void init();

    virtual void updateMapping();
protected:
    void addTopologyChangeToSpecificTopology(const TopologyChange *e);

};

} // namespace Core

} // namespace Sofa

#endif
