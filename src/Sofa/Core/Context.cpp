#include "Context.h"

namespace Sofa
{
namespace Core
{

Context::Context()
{
    setLocalFrame(getDefault()->getLocalFrame());
    setWorldGravity(getDefault()->getLocalGravity());
    setSpatialVelocity(getDefault()->getSpatialVelocity());
    setVelocityBasedLinearAcceleration(getDefault()->getVelocityBasedLinearAcceleration());
    setDt(getDefault()->getDt());
    setTime(getDefault()->getTime());
    setAnimate(getDefault()->getAnimate());
    setShowCollisionModels(getDefault()->getShowCollisionModels());
    setShowBoundingCollisionModels(getDefault()->getShowBoundingCollisionModels());
    setShowBehaviorModels(getDefault()->getShowBehaviorModels());
    setShowVisualModels(getDefault()->getShowVisualModels());
    setShowMappings(getDefault()->getShowMappings());
    setShowMechanicalMappings(getDefault()->getShowMechanicalMappings());
    setShowForceFields(getDefault()->getShowForceFields());
    setShowInteractionForceFields(getDefault()->getShowInteractionForceFields());
    setShowWireFrame(getDefault()->getShowWireFrame());
    setShowNormals(getDefault()->getShowNormals());
    setMultiThreadSimulation(getDefault()->getMultiThreadSimulation());
}

/// Projection from the local coordinate system to the world coordinate system.
const Context::Frame& Context::getLocalFrame() const { return localFrame_; }
/// Projection from the local coordinate system to the world coordinate system.
void Context::setLocalFrame(const Frame& f) { localFrame_ = f; }

/// Spatial velocity (linear, angular) of the local frame with respect to the world
const Context::SpatialVector& Context::getSpatialVelocity() const { return spatialVelocity_; }
/// Spatial velocity (linear, angular) of the local frame with respect to the world
void Context::setSpatialVelocity(const SpatialVector& v) { spatialVelocity_ = v; }

/// Linear acceleration of the origin induced by the angular velocity of the ancestors
const Context::Vec3& Context::getVelocityBasedLinearAcceleration() const { return velocityBasedLinearAcceleration_; }
/// Linear acceleration of the origin induced by the angular velocity of the ancestors
void Context::setVelocityBasedLinearAcceleration(const Vec3& a ) { velocityBasedLinearAcceleration_ = a; }



/// Simulation timestep
double Context::getDt() const
{
    return dt_;
}

/// Simulation time
double Context::getTime() const
{
    return time_;
}

/// Gravity vector in local coordinates
// const Context::Vec3& Context::getGravity() const
// {
// 	return gravity_;
// }

/// Gravity vector in world coordinates
Context::Vec3 Context::getLocalGravity() const
{
    return getLocalFrame().backProjectVector(worldGravity_);
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

/// Display flags: Bounding Collision Models
bool Context::getShowBoundingCollisionModels() const
{
    return showBoundingCollisionModels_;
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

/// Display flags: Mechanical Mappings
bool Context::getShowMechanicalMappings() const
{
    return showMechanicalMappings_;
}

/// Display flags: ForceFields
bool Context::getShowForceFields() const
{
    return showForceFields_;
}

/// Display flags: InteractionForceFields
bool Context::getShowInteractionForceFields() const
{
    return showInteractionForceFields_;
}

/// Display flags: WireFrame
bool Context::getShowWireFrame() const
{
    return showWireFrame_;
}

/// Display flags: Normal
bool Context::getShowNormals() const
{
    return showNormals_;
}

//===============================================================================

/// Simulation timestep
void Context::setDt(double val)
{
    dt_ = val;
}

/// Simulation time
void Context::setTime(double val)
{
    time_ = val;
}

/// Gravity vector
// void Context::setGravity(const Vec3& g)
// {
// 	gravity_ = g;
// }

/// Gravity vector
void Context::setWorldGravity(const Vec3& g)
{
    worldGravity_ = g;
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

/// Display flags: Bounding Collision Models
void Context::setShowBoundingCollisionModels(bool val)
{
    showBoundingCollisionModels_ = val;
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

/// Display flags: Mechanical Mappings
void Context::setShowMechanicalMappings(bool val)
{
    showMechanicalMappings_ = val;
}

/// Display flags: ForceFields
void Context::setShowForceFields(bool val)
{
    showForceFields_ = val;
}

/// Display flags: InteractionForceFields
void Context::setShowInteractionForceFields(bool val)
{
    showInteractionForceFields_ = val;
}

/// Display flags: WireFrame
void Context::setShowWireFrame(bool val)
{
    showWireFrame_ = val;
}

/// Display flags: Normals
void Context::setShowNormals(bool val)
{
    showNormals_ = val;
}



void Context::copyContext(const Context& c)
{
    *static_cast<ContextData*>(this) = *static_cast<const ContextData *>(&c);
}

using std::endl;

std::ostream& operator << (std::ostream& out, const Sofa::Core::Context& c )
{
    out<<endl<<"local gravity = "<<c.getLocalGravity();
    out<<endl<<"transform from local to world = "<<c.getLocalFrame();
    //out<<endl<<"transform from world to local = "<<c.getWorldToLocal();
    out<<endl<<"spatial velocity = "<<c.getSpatialVelocity();
    out<<endl<<"acceleration of the origin = "<<c.getVelocityBasedLinearAcceleration();
    return out;
}

} // namespace Core

} // namespace Sofa
