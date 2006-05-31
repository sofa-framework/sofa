#include "Context.h"

namespace Sofa
{
namespace Core
{

Context::Context()
{
    setLocalToWorld(getDefault()->getLocalToWorldTranslation(), getDefault()->getLocalToWorldRotationQuat(), getDefault()->getLocalToWorldRotationMatrix());
    setGravity(getDefault()->getGravity());
    setLinearVelocity(getDefault()->getLinearVelocity());
    setAngularVelocity(getDefault()->getAngularVelocity());
    setLinearAcceleration(getDefault()->getLinearAcceleration());
    setDt(getDefault()->getDt());
    setAnimate(getDefault()->getAnimate());
    setShowCollisionModels(getDefault()->getShowCollisionModels());
    setShowBehaviorModels(getDefault()->getShowBehaviorModels());
    setShowVisualModels(getDefault()->getShowVisualModels());
    setShowMappings(getDefault()->getShowMappings());
    setShowForceFields(getDefault()->getShowForceFields());
    setMultiThreadSimulation(getDefault()->getMultiThreadSimulation());
}


/// Simulation timestep
double Context::getDt() const
{
    return dt_;
}

/// Gravity vector in local coordinates
const double* Context::getGravity() const
{
    return gravity_;
}

/// Projection from the local coordinate system to the world coordinate system: translation part.
/// Returns a pointer to 3 doubles
const double* Context::getLocalToWorldTranslation() const
{
    return localToWorldTranslation_;
}

/// Projection from the local coordinate system to the world coordinate system: rotation part.
/// Returns a pointer to a 3x3 matrix (9 doubles, row-major format)
const double* Context::getLocalToWorldRotationMatrix() const
{
    return localToWorldRotationMatrix_;
}

/// Projection from the local coordinate system to the world coordinate system: rotation part.
/// Returns a pointer to a quaternion (4 doubles, <x,y,z,w> )
const double* Context::getLocalToWorldRotationQuat() const
{
    return localToWorldRotationQuat_;
}

/// Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame.
/// Returns a pointer to 3 doubles
const double* Context::getLinearAcceleration() const
{
    return linearAcceleration_;
}

/// Velocity of the local frame in the world coordinate system. The linear velocity is expressed at the origin of the world coordinate system.
/// Returns a pointer to 3 doubles
const double* Context::getLinearVelocity() const
{
    return linearVelocity_;
}

/// Velocity of the local frame in the world coordinate system.
/// Returns a pointer to 3 doubles
const double* Context::getAngularVelocity() const
{
    return angularVelocity_;
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
void Context::setDt(double val)
{
    dt_ = val;
}

/// Gravity vector as a pointer to 3 double
void Context::setGravity(const double* g)
{
    std::copy(g, g+3, gravity_);
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

void Context::setLocalToWorld( const double* translation, const double* rotationQuat, const double* rotationMatrix )
{
    std::copy (translation, translation+3, localToWorldTranslation_);
    std::copy (rotationQuat, rotationQuat+4, localToWorldRotationQuat_);
    std::copy (rotationMatrix, rotationMatrix+9, localToWorldRotationMatrix_);
}

void Context::setLinearVelocity( const double* v )
{
    std::copy(v, v+3, linearVelocity_);
}

void Context::setAngularVelocity( const double* v )
{
    std::copy(v, v+3, angularVelocity_);
}

/// Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame
void Context::setLinearAcceleration( const double* a )
{
    std::copy(a, a+3, linearAcceleration_);
}

void Context::copyContext(const Context& c)
{
    *static_cast<ContextData*>(this) = *static_cast<const ContextData *>(&c);
}

using std::endl;

std::ostream& operator << (std::ostream& out, const Sofa::Core::Context& c )
{
    out<<endl<<"gravity = "<<c.getGravity();
    out<<endl<<"transform from local to world = "<<c.getLocalToWorldTranslation()<<"   "<<c.getLocalToWorldRotationQuat();
    //out<<endl<<"transform from world to local = "<<c.getWorldToLocal();
    out<<endl<<"spatial velocity = "<<c.getLinearVelocity()<<"   "<<c.getAngularVelocity();
    out<<endl<<"acceleration of the origin = "<<c.getLinearAcceleration();
    return out;
}

} // namespace Core

} // namespace Sofa
