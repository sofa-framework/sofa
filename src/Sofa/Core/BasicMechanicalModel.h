#ifndef SOFA_CORE_BASICMECHANICALMODEL_H
#define SOFA_CORE_BASICMECHANICALMODEL_H

#include "ForceField.h"
#include "BasicMapping.h"
#include "Topology.h"
#include "Sofa/Abstract/Base.h"
#include "Sofa/Abstract/BehaviorModel.h"

namespace Sofa
{

namespace Core
{

class BasicMechanicalMapping;

class BasicMechanicalModel : public virtual Abstract::Base
{
public:
    virtual ~BasicMechanicalModel() { }

    // Basic Mappings are no longer handled by Mechanical Objects
    //virtual void addMapping(Core::BasicMapping *mMap) = 0;
    virtual void setMapping(Core::BasicMechanicalMapping *mMap) = 0;

    virtual void addMechanicalModel(Core::BasicMechanicalModel *mm) = 0;

    virtual void addForceField(Core::ForceField *mFField) = 0;

    virtual void resize(int vsize) = 0;

    virtual void init() = 0;

    virtual void beginIteration(double dt) = 0;

    virtual void endIteration(double dt) = 0;

    virtual void propagateX() = 0;

    virtual void propagateV() = 0;

    virtual void propagateDx() = 0;

    virtual void resetForce() = 0;

    virtual void accumulateForce() = 0;

    virtual void accumulateDf() = 0;

    virtual void applyConstraints() = 0;

    /// Set the behavior object currently ownning this model
    virtual void setObject(Abstract::BehaviorModel* obj) = 0;

    virtual void setTopology(Topology* topo) = 0;

    virtual Topology* getTopology() = 0;
};

} // namespace Core

} // namespace Sofa

#endif
