#ifndef SOFA_CORE_BASICMECHANICALMODEL_H
#define SOFA_CORE_BASICMECHANICALMODEL_H

#include "Topology.h"
#include "Sofa/Abstract/BaseObject.h"

namespace Sofa
{

namespace Core
{

class BasicMechanicalMapping;

class BasicMechanicalModel : public virtual Abstract::BaseObject
{
public:
    virtual ~BasicMechanicalModel() { }

    virtual void resize(int vsize) = 0;

    virtual void init() = 0;

    virtual void setTopology(Topology* topo) = 0;

    virtual Topology* getTopology() = 0;

    virtual void beginIteration(double dt) = 0;

    virtual void endIteration(double dt) = 0;

    virtual void propagateX() = 0;

    virtual void propagateV() = 0;

    virtual void propagateDx() = 0;

    virtual void resetForce() = 0;

    virtual void accumulateForce() = 0;

    virtual void accumulateDf() = 0;

    virtual void applyConstraints() = 0;
};

} // namespace Core

} // namespace Sofa

#endif
