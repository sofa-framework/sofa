#ifndef SOFA_CORE_BASICMECHANICALMODEL_H
#define SOFA_CORE_BASICMECHANICALMODEL_H

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

    /// @name Integration related methods
    /// @{

    virtual void beginIntegration(double /*dt*/) { }

    virtual void endIntegration(double /*dt*/) { }

    virtual void resetForce() =0;//{ vOp( VecId::force() ); }

    virtual void accumulateForce() { }

    virtual void accumulateDf() { }

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

};

} // namespace Core

} // namespace Sofa

#endif
