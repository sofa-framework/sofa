/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

namespace sofa
{

namespace core
{

namespace objectmodel
{

BaseContext::BaseContext()
{}

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

/// The Context is active
bool BaseContext::isActive() const { return true; }

/// The Context is not sleeping by default
bool BaseContext::isSleeping() const { return false; }

/// The Context can not change its sleeping state by default
bool BaseContext::canChangeSleepingState() const { return false; }


/// Gravity in the world coordinate system
const BaseContext::Vec3& BaseContext::getGravity() const
{
    static const Vec3 G(SReal(0),SReal(-9.81), SReal(0));
    return G;
}

/// Simulation timestep
SReal BaseContext::getDt() const
{
    return 0.01;
}

/// Simulation time
SReal BaseContext::getTime() const
{
    return 0.0;
}

/// Animation flag
bool BaseContext::getAnimate() const
{
    return true;
}


BaseContext* BaseContext::getRootContext() const
{
    return const_cast<BaseContext*>(this);
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
core::topology::BaseMeshTopology* BaseContext::getMeshTopology() const
{
    return this->get<sofa::core::topology::BaseMeshTopology>();
}

/// Mesh Topology that is local to this context (i.e. not within parent contexts)
core::topology::BaseMeshTopology* BaseContext::getLocalMeshTopology() const
{
    return this->get<sofa::core::topology::BaseMeshTopology>(Local);
}

/// Mesh Topology that is relevant for this context
/// (within it or its parents until a mapping is reached that does not preserve topologies).
core::topology::BaseMeshTopology* BaseContext::getActiveMeshTopology() const
{
    return this->get<sofa::core::topology::BaseMeshTopology>(Local);
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

BaseContext::GetObjectsCallBack::GetObjectsCallBack() {}
BaseContext::GetObjectsCallBack::~GetObjectsCallBack() {}

} // namespace objectmodel

} // namespace core

} // namespace sofa

