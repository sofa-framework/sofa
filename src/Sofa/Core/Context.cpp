#include "Context.h"
// #include "Sofa/Components/Graph/MechanicalAction.h"
// #include "Sofa/Components/Graph/MechanicalVPrintAction.h"

namespace Sofa
{
namespace Core
{

Context::Context()
    : worldGravity_(dataField(&worldGravity_, Vec3(0,0,0),"gravity","Gravity in the world coordinate system"))
    , dt_(dataField(&dt_,0.01,"dt","Time step"))
    , time_(dataField(&time_,0.,"time","Current time"))
    , animate_(dataField(&animate_,false,"animate","???"))
    , showCollisionModels_(dataField(&showCollisionModels_,false,"showCollisionModels","display flag"))
    , showBoundingCollisionModels_( dataField(&showBoundingCollisionModels_,false,"showBoundingCollisionModels","display flag"))
    , showBehaviorModels_(dataField(&showBehaviorModels_,false,"showBehaviorModels","display flag"))
    , showVisualModels_(dataField(&showVisualModels_,false,"showVisualModels","display flag"))
    , showMappings_(dataField(&showMappings_,false,"showMappings","display flag"))
    , showMechanicalMappings_(dataField(&showMechanicalMappings_,false,"showMechanicalMappings","display flag"))
    , showForceFields_(dataField(&showForceFields_,false,"showForceFields","display flag"))
    , showInteractionForceFields_(dataField(&showInteractionForceFields_,false,"showInteractionForceFields","display flag"))
    , showWireFrame_(dataField(&showWireFrame_,false,"showWireFrame","display flag"))
    , showNormals_(dataField(&showNormals_,false,"showNormals","display flag"))
    , multiThreadSimulation_(dataField(&multiThreadSimulation_,false,"multiThreadSimulation","Apply multithreaded simulation"))
{
    setPositionInWorld(Abstract::BaseContext::getPositionInWorld());
    setGravityInWorld(Abstract::BaseContext::getLocalGravity());
    setVelocityInWorld(Abstract::BaseContext::getVelocityInWorld());
    setVelocityBasedLinearAccelerationInWorld(Abstract::BaseContext::getVelocityBasedLinearAccelerationInWorld());
    setDt(Abstract::BaseContext::getDt());
    setTime(Abstract::BaseContext::getTime());
    setAnimate(Abstract::BaseContext::getAnimate());
    setShowCollisionModels(Abstract::BaseContext::getShowCollisionModels());
    setShowBoundingCollisionModels(Abstract::BaseContext::getShowBoundingCollisionModels());
    setShowBehaviorModels(Abstract::BaseContext::getShowBehaviorModels());
    setShowVisualModels(Abstract::BaseContext::getShowVisualModels());
    setShowMappings(Abstract::BaseContext::getShowMappings());
    setShowMechanicalMappings(Abstract::BaseContext::getShowMechanicalMappings());
    setShowForceFields(Abstract::BaseContext::getShowForceFields());
    setShowInteractionForceFields(Abstract::BaseContext::getShowInteractionForceFields());
    setShowWireFrame(Abstract::BaseContext::getShowWireFrame());
    setShowNormals(Abstract::BaseContext::getShowNormals());
    setMultiThreadSimulation(Abstract::BaseContext::getMultiThreadSimulation());
}

// Abstract::BaseContext* Context::getDefault()
// {
//     static Context defaultContext;
//     return &defaultContext;
// }


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



/// Simulation timestep
double Context::getDt() const
{
    return dt_.getValue();
}

/// Simulation time
double Context::getTime() const
{
    return time_.getValue();
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

/// Gravity vector in world coordinates
const Context::Vec3& Context::getGravityInWorld() const
{
    return worldGravity_.getValue();
}



/// Animation flag
bool Context::getAnimate() const
{
    return animate_.getValue();
}

/// MultiThreading activated
bool Context::getMultiThreadSimulation() const
{
    return multiThreadSimulation_.getValue();
}

/// Display flags: Collision Models
bool Context::getShowCollisionModels() const
{
    return showCollisionModels_.getValue();
}

/// Display flags: Bounding Collision Models
bool Context::getShowBoundingCollisionModels() const
{
    return showBoundingCollisionModels_.getValue();
}

/// Display flags: Behavior Models
bool Context::getShowBehaviorModels() const
{
    return showBehaviorModels_.getValue();
}

/// Display flags: Visual Models
bool Context::getShowVisualModels() const
{
    return showVisualModels_.getValue();
}

/// Display flags: Mappings
bool Context::getShowMappings() const
{
    return showMappings_.getValue();
}

/// Display flags: Mechanical Mappings
bool Context::getShowMechanicalMappings() const
{
    return showMechanicalMappings_.getValue();
}

/// Display flags: ForceFields
bool Context::getShowForceFields() const
{
    return showForceFields_.getValue();
}

/// Display flags: InteractionForceFields
bool Context::getShowInteractionForceFields() const
{
    return showInteractionForceFields_.getValue();
}

/// Display flags: WireFrame
bool Context::getShowWireFrame() const
{
    return showWireFrame_.getValue();
}

/// Display flags: Normal
bool Context::getShowNormals() const
{
    return showNormals_.getValue();
}

//===============================================================================

/// Simulation timestep
void Context::setDt(double val)
{
    dt_.setValue(val);
}

/// Simulation time
void Context::setTime(double val)
{
    time_.setValue(val);
}

/// Gravity vector
// void Context::setGravity(const Vec3& g)
// {
// 	gravity_ = g;
// }

/// Gravity vector
void Context::setGravityInWorld(const Vec3& g)
{
    worldGravity_ .setValue(g);
}

/// Animation flag
void Context::setAnimate(bool val)
{
    animate_.setValue(val);
}

/// MultiThreading activated
void Context::setMultiThreadSimulation(bool val)
{
    multiThreadSimulation_.setValue(val);
}

/// Display flags: Collision Models
void Context::setShowCollisionModels(bool val)
{
    showCollisionModels_.setValue(val);
}

/// Display flags: Bounding Collision Models
void Context::setShowBoundingCollisionModels(bool val)
{
    showBoundingCollisionModels_.setValue(val);
}

/// Display flags: Behavior Models
void Context::setShowBehaviorModels(bool val)
{
    showBehaviorModels_.setValue(val);
}

/// Display flags: Visual Models
void Context::setShowVisualModels(bool val)
{
    showVisualModels_.setValue(val);
}

/// Display flags: Mappings
void Context::setShowMappings(bool val)
{
    showMappings_.setValue(val);
}

/// Display flags: Mechanical Mappings
void Context::setShowMechanicalMappings(bool val)
{
    showMechanicalMappings_.setValue(val);
}

/// Display flags: ForceFields
void Context::setShowForceFields(bool val)
{
    showForceFields_.setValue(val);
}

/// Display flags: InteractionForceFields
void Context::setShowInteractionForceFields(bool val)
{
    showInteractionForceFields_.setValue(val);
}

/// Display flags: WireFrame
void Context::setShowWireFrame(bool val)
{
    showWireFrame_.setValue(val);
}

/// Display flags: Normals
void Context::setShowNormals(bool val)
{
    showNormals_.setValue(val);
}



void Context::copyContext(const Context& c)
{
    // *static_cast<ContextData*>(this) = *static_cast<const ContextData *>(&c);

    // BUGFIX 12/01/06 (Jeremie A.): Can't use operator= on the class as it will copy other data in the BaseContext class (such as name)...
    // *this = c;

    worldGravity_ = c.worldGravity_;  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
    dt_ = c.dt_;
    time_ = c.time_;
    animate_ = c.animate_;
    showCollisionModels_ = c.showCollisionModels_;
    showBoundingCollisionModels_ = c.showBoundingCollisionModels_;
    showBehaviorModels_ = c.showBehaviorModels_;
    showVisualModels_ = c.showVisualModels_;
    showMappings_ = c.showMappings_;
    showMechanicalMappings_ = c.showMechanicalMappings_;
    showForceFields_ = c.showForceFields_;
    showInteractionForceFields_ = c.showInteractionForceFields_;
    showWireFrame_ = c.showWireFrame_;
    showNormals_ = c.showNormals_;
    multiThreadSimulation_ = c.multiThreadSimulation_;

    localFrame_ = c.localFrame_;
    spatialVelocityInWorld_ = c.spatialVelocityInWorld_;
    velocityBasedLinearAccelerationInWorld_ = c.velocityBasedLinearAccelerationInWorld_;
}

using std::endl;

std::ostream& operator << (std::ostream& out, const Sofa::Core::Context& c )
{
    out<<endl<<"local gravity = "<<c.getLocalGravity();
    out<<endl<<"transform from local to world = "<<c.getPositionInWorld();
    //out<<endl<<"transform from world to local = "<<c.getWorldToLocal();
    out<<endl<<"spatial velocity = "<<c.getVelocityInWorld();
    out<<endl<<"acceleration of the origin = "<<c.getVelocityBasedLinearAccelerationInWorld();
    return out;
}




} // namespace Core

} // namespace Sofa



