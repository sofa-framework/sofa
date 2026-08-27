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
#include <sofa/core/objectmodel/Context.h>

namespace sofa::core::objectmodel
{

// Context constructor - BaseContext constructor already initializes all Data members
Context::Context() = default;

/// The Context is active
bool Context::isActive() const { return BaseContext::isActive(); }

/// State of the context
void Context::setActive(bool val) { BaseContext::setActive(val); }

/// The Context is sleeping
bool Context::isSleeping() const { return BaseContext::isSleeping(); }

/// Sleeping state of the context
void Context::setSleeping(bool val) { BaseContext::setSleeping(val); }

/// The Context can change its sleeping state
bool Context::canChangeSleepingState() const { return BaseContext::canChangeSleepingState(); }

/// Sleeping state change of the context
void Context::setChangeSleepingState(bool val) { BaseContext::setChangeSleepingState(val); }

/// Simulation timestep
SReal Context::getDt() const { return BaseContext::getDt(); }

/// Simulation time
SReal Context::getTime() const { return BaseContext::getTime(); }

/// Gravity vector in world coordinates
const Context::Vec3& Context::getGravity() const { return BaseContext::getGravity(); }

/// Animation flag
bool Context::getAnimate() const { return BaseContext::getAnimate(); }

/// Simulation timestep
void Context::setDt(SReal dt) { BaseContext::setDt(dt); }

/// Simulation time
void Context::setTime(SReal t) { BaseContext::setTime(t); }

/// Gravity vector
void Context::setGravity(const Vec3& g) { BaseContext::setGravity(g); }

/// Animation flag
void Context::setAnimate(bool val) { BaseContext::setAnimate(val); }

/// Display flags: Gravity
void Context::setDisplayWorldGravity(bool val) { BaseContext::setDisplayWorldGravity(val); }

//======================
void Context::copyContext(const Context& c) { BaseContext::copyContext(c); }

void Context::copySimulationContext(const Context& c) { BaseContext::copySimulationContext(c); }

// Additional compatibility: allow copying from BaseContext
void Context::copyContext(const BaseContext& c) { BaseContext::copyContext(c); }
void Context::copySimulationContext(const BaseContext& c) { BaseContext::copySimulationContext(c); }
} // namespace sofa::core::objectmodel

