/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
// #include <sofa/simulation/Visitor.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

Context::Context()
    : is_activated(initData(&is_activated, true, "activated", "To Activate a node"))
    , worldGravity_(initData(&worldGravity_, Vec3((SReal)0,(SReal)-9.81,(SReal)0),"gravity","Gravity in the world coordinate system"))
    , dt_(initData(&dt_,(SReal)0.01,"dt","Time step"))
    , time_(initData(&time_,(SReal)0.,"time","Current time"))
    , animate_(initData(&animate_,false,"animate","Animate the Simulation(applied at initialization only)"))
	, d_isSleeping(initData(&d_isSleeping, false, "sleeping", "The node is sleeping, and thus ignored by visitors."))
	, d_canChangeSleepingState(initData(&d_canChangeSleepingState, false, "canChangeSleepingState", "The node can change its sleeping state."))
#ifdef SOFA_SUPPORT_MULTIRESOLUTION
    , currentLevel_(initData(&currentLevel_,0,"currentLevel","Current level of details"))
    , coarsestLevel_(initData(&coarsestLevel_,3,"coarsestLevel","Coarsest level of details"))
    , finestLevel_(initData(&finestLevel_,0,"finestLevel","Finest level of details"))
#endif
{
#ifdef SOFA_SUPPORT_MOVING_FRAMES
    setPositionInWorld(objectmodel::BaseContext::getPositionInWorld());
    setGravity(objectmodel::BaseContext::getLocalGravity());
    setVelocityInWorld(objectmodel::BaseContext::getVelocityInWorld());
    setVelocityBasedLinearAccelerationInWorld(objectmodel::BaseContext::getVelocityBasedLinearAccelerationInWorld());
#endif

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

#ifdef SOFA_SUPPORT_MOVING_FRAMES
/// Projection from the local coordinate system to the world coordinate system.
const Context::Frame& Context::getPositionInWorld() const
{
    return localFrame_;
}
/// Projection from the local coordinate system to the world coordinate system.
void Context::setPositionInWorld(const Frame& f)
{
    localFrame_ = f;
}

/// Spatial velocity (linear, angular) of the local frame with respect to the world
const Context::SpatialVector& Context::getVelocityInWorld() const
{
    return spatialVelocityInWorld_;
}
/// Spatial velocity (linear, angular) of the local frame with respect to the world
void Context::setVelocityInWorld(const SpatialVector& v)
{
    spatialVelocityInWorld_ = v;
}

/// Linear acceleration of the origin induced by the angular velocity of the ancestors
const Context::Vec3& Context::getVelocityBasedLinearAccelerationInWorld() const
{
    return velocityBasedLinearAccelerationInWorld_;
}
/// Linear acceleration of the origin induced by the angular velocity of the ancestors
void Context::setVelocityBasedLinearAccelerationInWorld(const Vec3& a )
{
    velocityBasedLinearAccelerationInWorld_ = a;
}
/// Gravity vector in local coordinates
// const Context::Vec3& Context::getGravity() const
// {
// 	return gravity_;
// }

/// Gravity vector in local coordinates
Context::Vec3 Context::getLocalGravity() const
{
    return getPositionInWorld().backProjectVector(worldGravity_.getValue());
}
#endif


/// Simulation timestep
SReal Context::getDt() const
{
//    cerr << "Context::getDt() is " << dt_.getValue() << endl;
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


#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MULTIRESOLUTION
// Multiresolution

int Context::getCurrentLevel() const
{
    return currentLevel_.getValue();
}
int Context::getCoarsestLevel() const
{
    return coarsestLevel_.getValue();
}
int Context::getFinestLevel() const
{
    return finestLevel_.getValue();
}
#endif
#endif // SOFA_DEV

//===============================================================================

/// Simulation timestep
void Context::setDt(SReal val)
{
//    cerr << "Context::setDt("<< val <<")" << endl;
    dt_.setValue(val);
}

/// Simulation time
void Context::setTime(SReal val)
{
    time_.setValue(val);
}

/// Gravity vector
// void Context::setGravity(const Vec3& g)
// {
// 	gravity_ = g;
// }

/// Gravity vector
void Context::setGravity(const Vec3& g)
{
    worldGravity_ .setValue(g);
}

/// Animation flag
void Context::setAnimate(bool val)
{
    animate_.setValue(val);
}


#ifdef SOFA_SUPPORT_MULTIRESOLUTION
// Multiresolution

bool Context::setCurrentLevel(int l)
{
    if( l > coarsestLevel_.getValue() )
    {
        currentLevel_.setValue(coarsestLevel_.getValue());
        return false;
    }
    else if( l < 0 /*finestLevel_.getValue()*/ )
    {
// 		currentLevel_.setValue(finestLevel_.getValue());
        currentLevel_.setValue( 0 );
        return false;
    }
    currentLevel_.setValue(l);
    if( l == coarsestLevel_.getValue() ) return false;
    return true;
}
void Context::setCoarsestLevel(int l)
{
    coarsestLevel_.setValue( l );
}
void Context::setFinestLevel(int l)
{
    finestLevel_.setValue( l );
}
#endif

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


#ifdef SOFA_SUPPORT_MOVING_FRAMES
    setPositionInWorld( c.getPositionInWorld());
    spatialVelocityInWorld_ = c.spatialVelocityInWorld_;
    velocityBasedLinearAccelerationInWorld_ = c.velocityBasedLinearAccelerationInWorld_;
#endif

#ifdef SOFA_SUPPORT_MULTIRESOLUTION
    // for multiresolution
// 	finestLevel_ = c.finestLevel_;
// 	coarsestLevel_ = c.coarsestLevel_;
// 	currentLevel_ = c.currentLevel_;
#endif


}




} // namespace objectmodel

} // namespace core

} // namespace sofa

