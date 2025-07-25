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

Context::Context()
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

/// The Context is active
bool Context::isActive() const {return is_activated.getValue();}

/// State of the context
void Context::setActive(bool val)
{
    is_activated.setValue(val);
}

/// The Context is sleeping
bool Context::isSleeping() const 
{
	return d_isSleeping.getValue();
}

/// Sleeping state of the context
void Context::setSleeping(bool val)
{
	d_isSleeping.setValue(val);
}

/// The Context can change its sleeping state
bool Context::canChangeSleepingState() const 
{ 
	return d_canChangeSleepingState.getValue(); 
}

/// Sleeping state change of the context
void Context::setChangeSleepingState(bool val)
{
	d_canChangeSleepingState.setValue(val);
}



/// Simulation timestep
SReal Context::getDt() const
{
    return dt_.getValue();
}

/// Simulation time
SReal Context::getTime() const
{
    return time_.getValue();
}

/// Gravity vector in world coordinates
const Context::Vec3& Context::getGravity() const
{
    return worldGravity_.getValue();
}

/// Animation flag
bool Context::getAnimate() const
{
    return animate_.getValue();
}

//===============================================================================

/// Simulation timestep
void Context::setDt(SReal dt)
{
    dt_.setValue(dt);
}

/// Simulation time
void Context::setTime(SReal t)
{
    time_.setValue(t);
}

/// Gravity vector
void Context::setGravity(const Vec3& g)
{
    worldGravity_ .setValue(g);
}

/// Animation flag
void Context::setAnimate(const bool val)
{
    animate_.setValue(val);
}

//======================
void Context::copyContext(const Context& c)
{
    // BUGFIX 12/01/06 (Jeremie A.): Can't use operator= on the class as it will copy other data in the BaseContext class (such as name)...
    // *this = c;

    copySimulationContext(c);
}


void Context::copySimulationContext(const Context& c)
{
    worldGravity_.setValue(c.getGravity());  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
    setDt(c.getDt());
    setTime(c.getTime());
    setAnimate(c.getAnimate());
}
} // namespace sofa::core::objectmodel

