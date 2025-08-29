/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/Shader.h>
#include <iostream>

namespace sofa::core::objectmodel
{

BaseContext::BaseContext()
    : is_activated(initData(&is_activated, true, "activated", "To Activate a node"))
    , worldGravity_(initData(&worldGravity_, Vec3(SReal(0),SReal(-9.81),SReal(0)),"gravity","Gravity in the world coordinate system"))
    , dt_(initData(&dt_,SReal(0.01),"dt","Time step"))
    , time_(initData(&time_,SReal(0.),"time","Current time"))
    , animate_(initData(&animate_,false,"animate","Animate the Simulation(applied at initialization only)"))
    , d_isSleeping(initData(&d_isSleeping, false, "sleeping", "The node is sleeping, and thus ignored by visitors."))
    , d_canChangeSleepingState(initData(&d_canChangeSleepingState, false, "canChangeSleepingState", "The node can change its sleeping state."))
{
    animate_.setReadOnly(true);
    dt_.setReadOnly(true);
    time_.setReadOnly(true);
}


BaseContext::~BaseContext()
{}

BaseContext* BaseContext::getDefault()
{
    static BaseContext defaultContext;
    return &defaultContext;
}

////////////////
// Parameters //
////////////////

/// State of the context
void BaseContext::setActive(bool val) { is_activated.setValue(val); }

/// The Context is active
bool BaseContext::isActive() const { return is_activated.getValue(); }


/// Sleeping state of the context
void BaseContext::setSleeping(bool val){ d_isSleeping.setValue(val); }

/// The Context is not sleeping by default
bool BaseContext::isSleeping() const { return d_isSleeping.getValue(); }

/// The Context can not change its sleeping state by default
bool BaseContext::canChangeSleepingState() const { return d_canChangeSleepingState.getValue(); }

/// Sleeping state change of the context
void BaseContext::setChangeSleepingState(bool val)
{
    d_canChangeSleepingState.setValue(val);
}

/// Gravity vector
void BaseContext::setGravity(const Vec3& g)
{
    worldGravity_ .setValue(g);
}

/// Gravity in the world coordinate system
const BaseContext::Vec3& BaseContext::getGravity() const
{
    return worldGravity_.getValue();
}

/// Simulation timestep
void BaseContext::setDt(SReal dt)
{
    dt_.setValue(dt);
}

/// Simulation timestep
SReal BaseContext::getDt() const
{
    return dt_.getValue();
}

/// Simulation time
void BaseContext::setTime(SReal t)
{
    time_.setValue(t);
}

/// Simulation time
SReal BaseContext::getTime() const
{
    return time_.getValue();
}

/// Animation flag
void BaseContext::setAnimate(const bool val)
{
    animate_.setValue(val);
}

/// Animation flag
bool BaseContext::getAnimate() const
{
    return animate_.getValue();
}

/// Display flags: Gravity
void BaseContext::setDisplayWorldGravity(bool val)
{
    worldGravity_.setDisplayed(val);
}

BaseContext* BaseContext::getRootContext() const
{
    return const_cast<BaseContext*>(this);
}

//======================
void BaseContext::copyContext(const BaseContext& c)
{
    // BUGFIX 12/01/06 (Jeremie A.): Can't use operator= on the class as it will copy other data in the BaseContext class (such as name)...
    // *this = c;
    copySimulationContext(c);
}


void BaseContext::copySimulationContext(const BaseContext& c)
{
    worldGravity_.setValue(c.getGravity());  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
    setDt(c.getDt());
    setTime(c.getTime());
    setAnimate(c.getAnimate());
}


////////////////
// Containers //
////////////////

/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BaseContext::getObject(const ClassInfo& /*class_info*/, SearchDirection /*dir*/) const
{
    msg_warning("calling unimplemented getObject method");
    return nullptr;
}

/// Generic object access, given a set of required tags, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BaseContext::getObject(const ClassInfo& /*class_info*/, const TagSet& /*tags*/, SearchDirection /*dir*/) const
{
    msg_warning("calling unimplemented getObject method");
    return nullptr;
}

/// Generic object access, given a path from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BaseContext::getObject(const ClassInfo& /*class_info*/, const std::string& /*path*/) const
{
    msg_warning("calling unimplemented getObject method");
    return nullptr;
}

/// Generic list of objects access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void BaseContext::getObjects(const ClassInfo& /*class_info*/, GetObjectsCallBack& /*container*/, SearchDirection /*dir*/) const
{
    msg_warning("calling unimplemented getObject method");
}

/// Generic list of objects access, given a set of required tags, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void BaseContext::getObjects(const ClassInfo& /*class_info*/, GetObjectsCallBack& /*container*/, const TagSet& /*tags*/, SearchDirection /*dir*/) const
{
    msg_error("calling unimplemented getObject method");
}

/// Degrees-of-Freedom
core::BaseState* BaseContext::getState() const
{
    return this->get<sofa::core::BaseState>();
}

/// Mechanical Degrees-of-Freedom
core::behavior::BaseMechanicalState* BaseContext::getMechanicalState() const
{
    return this->get<sofa::core::behavior::BaseMechanicalState>();
}

/// Mass
behavior::BaseMass* BaseContext::getMass() const
{
    return this->get<sofa::core::behavior::BaseMass>();
}


/// Topology
core::topology::Topology* BaseContext::getTopology() const
{
    return this->get<sofa::core::topology::Topology>();
}

/// Mesh Topology (unified interface for both static and dynamic topologies)
core::topology::BaseMeshTopology* BaseContext::getMeshTopology(SearchDirection dir) const
{
    return this->get<sofa::core::topology::BaseMeshTopology>(dir);
}

core::topology::BaseMeshTopology* BaseContext::getMeshTopologyLink(SearchDirection dir) const
{
    return this->get<sofa::core::topology::BaseMeshTopology>(dir);
}

/// Shader
core::visual::Shader* BaseContext::getShader() const
{
    return this->get<sofa::core::visual::Shader>();
}

/// Propagate an event
void BaseContext::propagateEvent( const core::ExecParams*, Event* )
{
    msg_warning("propagateEvent not overloaded, does nothing");
}

void BaseContext::executeVisitor(simulation::Visitor*, bool)
{
    msg_warning("executeVisitor not overloaded, does nothing");
}

std::ostream& operator << (std::ostream& out, const BaseContext&)
{
    return out;
}

void BaseContext::notifyAddSlave(core::objectmodel::BaseObject* /*master*/, core::objectmodel::BaseObject* /*slave*/)
{
}

void BaseContext::notifyRemoveSlave(core::objectmodel::BaseObject* /*master*/, core::objectmodel::BaseObject* /*slave*/)
{
}

void BaseContext::notifyMoveSlave(core::objectmodel::BaseObject* /*previousMaster*/, core::objectmodel::BaseObject* /*master*/, core::objectmodel::BaseObject* /*slave*/)
{
}

} // namespace sofa::core::objectmodel
