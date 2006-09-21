#ifndef SOFA_CORE_BASICTOPOLOGY_H
#define SOFA_CORE_BASICTOPOLOGY_H

#include <stdlib.h>
#include <vector>
#include <list>
#include <string>
#include <iostream>
#include "Sofa/Abstract/BaseObject.h"

namespace Sofa
{

namespace Core
{
// forward definition
class TopologyContainer;
class TopologyModifier;
class TopologyAlgorithms;
class GeometryAlgorithms;
class TopologicalMapping;


/** base class to indicate which topology change occured */
class TopologyChange
{
protected:
    unsigned int changeType; /// a code that tells the nature of the Topology modification event (could be an enum)
public:
    unsigned int getChangeType() const
    {
        return changeType;
    }
};

/** Base class that gives access to 5 topology related objects and
an array of topology modification */
class BasicTopology : public virtual Abstract::BaseObject
{
protected :

    TopologyContainer *topologyContainerObject;
    TopologyModifier *topologyModifierObject;
    TopologyAlgorithms *topologyAlgorithmsObject;
    GeometryAlgorithms *geometryAlgorithmsObject;
    /** array of topology modification that have already occured (addition) or
    will occur (deletion) */
    std::list<const TopologyChange *> changeList;
    bool mainTopology;
private:
    void addTopologyChange(const TopologyChange *e)
    {
        changeList.push_back(e);
    }
public :
    std::list<const TopologyChange *>::const_iterator firstChange() const
    {
        return changeList.begin();
    }
    std::list<const TopologyChange *>::const_iterator lastChange() const
    {
        return changeList.end();
    }

    TopologyContainer *getTopologyContainer() const
    {
        return topologyContainerObject;
    }
    /// only a mainTopology can be modified (thus returns 0 if not a main topologY)
    TopologyAlgorithms *getTopologyAlgorithms() const
    {
        if (mainTopology)
            return topologyAlgorithmsObject;
        else
            return (TopologyAlgorithms *) 0;
    }
    GeometryAlgorithms *getGeometryAlgorithms() const
    {
        return geometryAlgorithmsObject;
    }
    BasicTopology(bool _isMainTopology=true) : topologyContainerObject(0),
        topologyModifierObject(0),
        topologyAlgorithmsObject(0),geometryAlgorithmsObject(0),
        mainTopology(_isMainTopology)
    {
    }
    ~BasicTopology();
    virtual void propagateTopologicalChanges()
    {
    }
    bool isMainTopology() const
    {
        return mainTopology;
    }
    friend class TopologyContainer;
    friend class TopologyModifier;
    friend class TopologyAlgorithms;
    friend class GeometryAlgorithms;
    friend class TopologicalMapping;

};
/** A class that contains a description of the topology (set of edges, triangles,...) */
class TopologyContainer
{
protected:
    BasicTopology *topology;
    void addTopologyChange(const TopologyChange *e)
    {
        topology->addTopologyChange(e);
    }
public:
    TopologyContainer(BasicTopology *_top) : topology(_top) {}
};
/** A class that contains a set of low-level methods that perform topological changes */
class TopologyModifier
{
protected:
    BasicTopology *topology;
    void addTopologyChange(const TopologyChange *e)
    {
        topology->addTopologyChange(e);
    }

public:
    TopologyModifier(BasicTopology *_top) : topology(_top) {}
};
/** A class that contains a set of high-level (user friendly) methods that perform topological changes */
class TopologyAlgorithms
{
protected:
    BasicTopology *topology;
    void addTopologyChange(const TopologyChange *e)
    {
        topology->addTopologyChange(e);
    }
public:
    TopologyAlgorithms(BasicTopology *_top) : topology(_top) {}
};
/** A class that contains a set of methods that describes the geometry of the object */
class GeometryAlgorithms
{
protected:
    BasicTopology *topology;
    void addTopologyChange(const TopologyChange *e)
    {
        topology->addTopologyChange(e);
    }
public:
    GeometryAlgorithms(BasicTopology *_top) : topology(_top) {}
};

} // namespace Core

} // namespace Sofa

#endif
