#include "Context.h"

namespace Sofa
{
namespace Core
{

/// Simulation timestep
float Context::getDt() const
{
    return dt_;
}

/// Gravity vector as a pointer to 3 double
const Context::Vec& Context::getGravity() const
{
    return gravity_;
}

const Context::Frame& Context::getLocalToWorld() const
{
    return localToWorld_;
}
const Context::Frame& Context::getWorldToLocal() const
{
    return worldToLocal_;
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

void Context::setWorldToLocal( const Frame& f )
{
    worldToLocal_ = f;
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

void Context::copyContextFrom( const Context* f )
{
    *this = *f;
}

Context Context::getDefault()
{
    Context d;
    d.setGravity( Vec(0,-9.81,0) );
    d.setLocalToWorld( Frame::identity() );
    d.setWorldToLocal( Frame::identity() );
    return d;
}



}//Core
}//Sofa

std::ostream& operator << (std::ostream& out, const Sofa::Core::Context& c )
{
    out<<endl<<"gravity = "<<c.getGravity();
    out<<endl<<"transform from local to world = "<<c.getLocalToWorld();
    out<<endl<<"transform from world to local = "<<c.getWorldToLocal();
    return out;
}


