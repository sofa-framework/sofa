#ifndef SOFA_CORE_BASICMECHANICALMODEL_H
#define SOFA_CORE_BASICMECHANICALMODEL_H

#include "Topology.h"
#include "Sofa/Abstract/BaseObject.h"
#include "Sofa/Core/Encoding.h"
#include <iostream>

namespace Sofa
{

namespace Core
{

using namespace Encoding;

class BasicMechanicalMapping;

class BasicMechanicalModel : public virtual Abstract::BaseObject
{
public:
    BasicMechanicalModel()
        : Abstract::BaseObject()
    {}
    virtual ~BasicMechanicalModel()
    { }

    virtual BasicMechanicalModel* resize(int vsize) = 0;

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

    /// @name Integration related methods
    /// @{

    virtual void vAlloc(VecId v) = 0; // {}

    virtual void vFree(VecId v) = 0; // {}

    virtual void vOp(VecId v, VecId a = VecId::null(), VecId b = VecId::null(), double f=1.0) = 0; // {}

    virtual double vDot(VecId a, VecId b) = 0; //{ return 0; }

    virtual void setX(VecId v) = 0; //{}

    virtual void setV(VecId v) = 0; //{}

    virtual void setF(VecId v) = 0; //{}

    virtual void setDx(VecId v) = 0; //{}

    /// @}

    /// @name Debug
    /// @{
    virtual void printDOF( VecId, std::ostream& =std::cerr ) = 0;
    /// @}

    ///  Default: do not change the context.
    virtual void updateContext( Context* )
    {}

}
;

} // namespace Core

} // namespace Sofa

#endif
