#ifndef SOFA_ABSTRACT_BASECONTEXT_H
#define SOFA_ABSTRACT_BASECONTEXT_H

#include "Base.h"
#include <Sofa/Components/Common/SolidTypes.h>

namespace Sofa
{

namespace Abstract
{

class BaseObject;

/// Base class for storing shared variables and parameters.
class BaseContext : public virtual Base
{
public:
    typedef Components::Common::SolidTypes<double> SolidTypes;
    typedef SolidTypes::Transform Frame;
    typedef SolidTypes::Vec Vec3;
    typedef SolidTypes::Rot Quat;
    typedef SolidTypes::Mat Mat33;
    typedef SolidTypes::SpatialVector SpatialVector;

    BaseContext();
    virtual ~BaseContext();

    static BaseContext* getDefault();

    /// @name Parameters
    /// @{

    /// Simulation time
    virtual double getTime() const;

    /// Simulation timestep
    virtual double getDt() const;

    /// Animation flag
    virtual bool getAnimate() const;

    /// MultiThreading activated
    virtual bool getMultiThreadSimulation() const;

    /// Display flags: Collision Models
    virtual bool getShowCollisionModels() const;

    /// Display flags: Behavior Models
    virtual bool getShowBehaviorModels() const;

    /// Display flags: Visual Models
    virtual bool getShowVisualModels() const;

    /// Display flags: Mappings
    virtual bool getShowMappings() const;

    /// Display flags: ForceFields
    virtual bool getShowForceFields() const;

    /// Display flags: WireFrame
    virtual bool getShowWireFrame() const;

    /// Display flags: Normals
    virtual bool getShowNormals() const;

    /// @}


    /// @name Local Coordinate System
    /// @{
    /// Projection from the local coordinate system to the world coordinate system.
    virtual const Frame& getLocalFrame() const;
    /// Projection from the local coordinate system to the world coordinate system.
    virtual void setLocalFrame(const Frame&) {}

    /// Spatial velocity (linear, angular) of the local frame with respect to the world
    virtual const SpatialVector& getSpatialVelocity() const;
    /// Spatial velocity (linear, angular) of the local frame with respect to the world
    virtual void setSpatialVelocity(const SpatialVector&) {}

    /// Linear acceleration of the origin induced by the angular velocity of the ancestors
    virtual const Vec3& getVelocityBasedLinearAcceleration() const;
    /// Linear acceleration of the origin induced by the angular velocity of the ancestors
    virtual void setVelocityBasedLinearAcceleration(const Vec3& ) {}
    /// @}


    /// Gravity in local coordinates
    virtual const Vec3& getGravity() const;
    /// Gravity in local coordinates
    virtual void setGravity( const Vec3& ) { }

    /// @name Variables
    /// @{

    /// Mechanical Degrees-of-Freedom
    virtual BaseObject* getMechanicalModel() const;

    /// Topology
    virtual BaseObject* getTopology() const;

    /// @}

    /// @name Parameters Setters
    /// @{


    /// Simulation timestep
    virtual void setDt( double /*dt*/ ) { }

    /// Animation flag
    virtual void setAnimate(bool /*val*/) { }

    /// MultiThreading activated
    virtual void setMultiThreadSimulation(bool /*val*/) { }

    /// Display flags: Collision Models
    virtual void setShowCollisionModels(bool /*val*/) { }

    /// Display flags: Behavior Models
    virtual void setShowBehaviorModels(bool /*val*/) { }

    /// Display flags: Visual Models
    virtual void setShowVisualModels(bool /*val*/) { }

    /// Display flags: Mappings
    virtual void setShowMappings(bool /*val*/) { }

    /// Display flags: ForceFields
    virtual void setShowForceFields(bool /*val*/) { }

    /// Display flags: WireFrame
    virtual void setShowWireFrame(bool /*val*/) { }

    /// Display flags: Normals
    virtual void setShowNormals(bool /*val*/) { }

    /// @}

    /// @name Variables Setters
    /// @{

    /// Mechanical Degrees-of-Freedom
    virtual void setMechanicalModel( Abstract::BaseObject* ) { }

    /// Topology
    virtual void setTopology( Abstract::BaseObject* ) { }

    /// @}

    /// @name Adding/Removing objects. Note that these methods can fail if the context don't support attached objects
    /// @{

    /// Add an object, or return false if not supported
    virtual bool addObject( BaseObject* /*obj*/ ) { return false; }

    /// Remove an object, or return false if not supported
    virtual bool removeObject( BaseObject* /*obj*/ ) { return false; }

    /// @}

};

} // namespace Abstract

} // namespace Sofa

#endif
