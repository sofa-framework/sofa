#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEMECHANICALSTATE_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEMECHANICALSTATE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/io/Encoding.h>
#include <iostream>

using namespace sofa::helper;

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

class BaseMechanicalMapping;

class BaseMechanicalState : public virtual objectmodel::BaseObject
{
public:
    BaseMechanicalState ()
        : objectmodel::BaseObject()
    {}
    virtual ~BaseMechanicalState ()
    { }

    virtual void resize(int vsize) = 0;

    virtual void init() = 0;

    /// @name Integration related methods
    /// @{

    virtual void beginIntegration(double /*dt*/) { }

    virtual void endIntegration(double /*dt*/) { }

    virtual void resetForce() =0;//{ vOp( helper::io::VecId::force() ); }

    virtual void resetConstraint() =0;

    virtual void accumulateForce() { }

    virtual void accumulateDf() { }

    virtual void vAlloc(helper::io::VecId v) = 0; // {}

    virtual void vFree(helper::io::VecId v) = 0; // {}

    virtual void vOp(helper::io::VecId v, helper::io::VecId a = helper::io::VecId::null(), helper::io::VecId b = helper::io::VecId::null(), double f=1.0) = 0; // {}

    virtual double vDot(helper::io::VecId a, helper::io::VecId b) = 0; //{ return 0; }

    virtual void setX(helper::io::VecId v) = 0; //{}

    virtual void setV(helper::io::VecId v) = 0; //{}

    virtual void setF(helper::io::VecId v) = 0; //{}

    virtual void setDx(helper::io::VecId v) = 0; //{}

    virtual void setC(helper::io::VecId v) = 0; //{}

    /// @}

    /// @name Debug
    /// @{
    virtual void printDOF( helper::io::VecId, std::ostream& =std::cerr ) = 0;
    /// @}


    /*! \fn void addBBox()
     *  \brief Used to add the bounding-box of this mechanical model to the given bounding box.
     *
     *  Note that if it does not make sense for this particular object (such as if the DOFs are not 3D), then the default implementation displays a warning message and returns false.
     */
    //virtual bool addBBox(double* /*minBBox*/, double* /*maxBBox*/)
    //{
    //  std::cerr << "warning: unumplemented method MechanicalState::addBBox() called.\n";
    //  return false;
    //}

};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
