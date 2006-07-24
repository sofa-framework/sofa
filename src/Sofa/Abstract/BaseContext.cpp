#include "BaseContext.h"
#include "BaseObject.h"

namespace Sofa
{

namespace Abstract
{

BaseContext::BaseContext()
{
}

BaseContext::~BaseContext()
{
}

BaseContext* BaseContext::getDefault()
{
    static BaseContext defaultContext;
    return &defaultContext;
}

////////////////
// Parameters //
////////////////

/// Gravity in the local coordinate system
const BaseContext::Vec3& BaseContext::getGravity() const
{
    static const Vec3 G(0,-9.81,0);
    return G;
}

/// Gravity in the world coordinate system
const BaseContext::Vec3& BaseContext::getWorldGravity() const
{
    return getGravity();
}

/// Simulation timestep
double BaseContext::getDt() const
{
    return 0.01;
}

/// Simulation time
double BaseContext::getTime() const
{
    return 0.0;
}

/// Animation flag
bool BaseContext::getAnimate() const
{
    return true;
}

/// MultiThreading activated
bool BaseContext::getMultiThreadSimulation() const
{
    return false;
}

/// Display flags: Collision Models
bool BaseContext::getShowCollisionModels() const
{
    return true;
}

/// Display flags: Behavior Models
bool BaseContext::getShowBehaviorModels() const
{
    return true;
}

/// Display flags: Visual Models
bool BaseContext::getShowVisualModels() const
{
    return true;
}

/// Display flags: Mappings
bool BaseContext::getShowMappings() const
{
    return true;
}

/// Display flags: ForceFields
bool BaseContext::getShowForceFields() const
{
    return true;
}

/// Display flags: WireFrame
bool BaseContext::getShowWireFrame() const
{
    return true;
}

/// Display flags: Normals
bool BaseContext::getShowNormals() const
{
    return true;
}


//////////////////////////////
// Local Coordinates System //
//////////////////////////////


/// Projection from the local coordinate system to the world coordinate system.
const BaseContext::Frame& BaseContext::getLocalFrame() const
{
    static const Frame f;
    return f;
}

/// Spatial velocity (linear, angular) of the local frame with respect to the world
const BaseContext::SpatialVector& BaseContext::getSpatialVelocity() const
{
    static const SpatialVector v( Vec3(0,0,0), Vec3(0,0,0) );
    return v;
}

/// Linear acceleration of the origin induced by the angular velocity of the ancestors
const BaseContext::Vec3& BaseContext::getVelocityBasedLinearAcceleration() const
{
    static const Vec3 a(0,0,0);
    return a;
}


///////////////
// Variables //
///////////////


/// Mechanical Degrees-of-Freedom
BaseObject* BaseContext::getMechanicalModel() const
{
    return NULL;
}

/// Topology
BaseObject* BaseContext::getTopology() const
{
    return NULL;
}

} // namespace Abstract

} // namespace Sofa
