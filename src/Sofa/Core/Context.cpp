#include "Context.h"

namespace Sofa
{
namespace Core
{

Context::Context()
    : gravity_( Vec(0,-9.81f,0) )
    , localToWorld_( Frame::identity() )
    , spatialVelocity_( Vec(0,0,0),Vec(0,0,0) )
    , originAcceleration_( Vec(0,0,0) )
    , dt_(0.04f)
    , animate_(false)
    , showCollisionModels_(false)
    , showBehaviorModels_(true)
    , showVisualModels_(true)
    , showMappings_(false)
    , showForceFields_(true)
    , multiThreadSimulation_(false)
{}


/// Simulation timestep
float Context::getDt() const
{
    return dt_;
}

/// Gravity vector in local coordinates
const Context::Vec Context::getGravity() const
{
    return getLocalToWorld().backProjectVector(gravity_);
}

const Context::Frame& Context::getLocalToWorld() const
{
    return localToWorld_;
}

const Context::SpatialVelocity& Context::getSpatialVelocity() const
{
    return spatialVelocity_;
}

/// Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame
const Context::Vec& Context::getOriginAcceleration() const
{
    return originAcceleration_;
}


Context::Vec Context::getLinearVelocity() const
{
    return spatialVelocity_.freeVec;
}
Context::Vec Context::getAngularVelocity() const
{
    return spatialVelocity_.lineVec;
}

/// Animation flag
bool Context::getAnimate() const
{
    return animate_;
}

/// MultiThreading activated
bool Context::getMultiThreadSimulation() const
{
    return multiThreadSimulation_;
}

/// Display flags: Collision Models
bool Context::getShowCollisionModels() const
{
    return showCollisionModels_;
}

/// Display flags: Behavior Models
bool Context::getShowBehaviorModels() const
{
    return showBehaviorModels_;
}

/// Display flags: Visual Models
bool Context::getShowVisualModels() const
{
    return showVisualModels_;
}

/// Display flags: Mappings
bool Context::getShowMappings() const
{
    return showMappings_;
}

/// Display flags: ForceFields
bool Context::getShowForceFields() const
{
    return showForceFields_;
}

//===============================================================================

/// Simulation timestep
void Context::setDt(float val)
{
    dt_ = val;
}

/// Gravity vector as a pointer to 3 double
void Context::setGravity(const Vec& g)
{
    gravity_ = g;
}

void Context::setLocalToWorld( const Frame& f )
{
    localToWorld_ = f;
}

void Context::setSpatialVelocity( const SpatialVelocity& v )
{
    spatialVelocity_ = v;
}

/// Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame
void Context::setOriginAcceleration( const Vec& a )
{
    originAcceleration_ = a;
}


/// Animation flag
void Context::setAnimate(bool val)
{
    animate_ = val;
}

/// MultiThreading activated
void Context::setMultiThreadSimulation(bool val)
{
    multiThreadSimulation_ = val;
}

/// Display flags: Collision Models
void Context::setShowCollisionModels(bool val)
{
    showCollisionModels_ = val;
}

/// Display flags: Behavior Models
void Context::setShowBehaviorModels(bool val)
{
    showBehaviorModels_ = val;
}

/// Display flags: Visual Models
void Context::setShowVisualModels(bool val)
{
    showVisualModels_ = val;
}

/// Display flags: Mappings
void Context::setShowMappings(bool val)
{
    showMappings_ = val;
}

/// Display flags: ForceFields
void Context::setShowForceFields(bool val)
{
    showForceFields_ = val;
}

Context Context::getDefault()
{
    Context d;
    return d;
}

std::ostream& operator << (std::ostream& out, const Sofa::Core::Context& c )
{
    out<<endl<<"gravity = "<<c.getGravity();
    out<<endl<<"transform from local to world = "<<c.getLocalToWorld();
    //out<<endl<<"transform from world to local = "<<c.getWorldToLocal();
    out<<endl<<"spatial velocity = "<<c.getSpatialVelocity();
    out<<endl<<"acceleration of the origin = "<<c.getOriginAcceleration();
    return out;
}


}//Core
}//Sofa


