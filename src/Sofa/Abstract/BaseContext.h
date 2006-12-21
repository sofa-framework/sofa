#ifndef SOFA_ABSTRACT_BASECONTEXT_H
#define SOFA_ABSTRACT_BASECONTEXT_H

#include "Base.h"
#include <Sofa/Components/Common/SolidTypes.h>
//#include <Sofa/Components/Common/SofaBaseMatrix.h>
//#include <Sofa/Components/Graph/Action.h>
#include "Sofa/Core/Encoding.h"
#include <set>

namespace Sofa
{

namespace Components
{
namespace Graph
{
class Action;
}
}

namespace Abstract
{

class BaseObject;
class Event;

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
    typedef Core::Encoding::VecId VecId;

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

    /// Display flags: Bounding Collision Models
    virtual bool getShowBoundingCollisionModels() const;

    /// Display flags: Behavior Models
    virtual bool getShowBehaviorModels() const;

    /// Display flags: Visual Models
    virtual bool getShowVisualModels() const;

    /// Display flags: Mappings
    virtual bool getShowMappings() const;

    /// Display flags: Mechanical Mappings
    virtual bool getShowMechanicalMappings() const;

    /// Display flags: ForceFields
    virtual bool getShowForceFields() const;

    /// Display flags: InteractionForceFields
    virtual bool getShowInteractionForceFields() const;

    /// Display flags: WireFrame
    virtual bool getShowWireFrame() const;

    /// Display flags: Normals
    virtual bool getShowNormals() const;

    /// Multiresolution
    virtual int getCurrentLevel() const;
    virtual int getCoarsestLevel() const;
    virtual int getFinestLevel() const;
    virtual unsigned int nbLevels() const;

    /// @}


    /// @name Local Coordinate System
    /// @{
    /// Projection from the local coordinate system to the world coordinate system.
    virtual const Frame& getPositionInWorld() const;
    /// Projection from the local coordinate system to the world coordinate system.
    virtual void setPositionInWorld(const Frame&)
    {}

    /// Spatial velocity (linear, angular) of the local frame with respect to the world
    virtual const SpatialVector& getVelocityInWorld() const;
    /// Spatial velocity (linear, angular) of the local frame with respect to the world
    virtual void setVelocityInWorld(const SpatialVector&)
    {}

    /// Linear acceleration of the origin induced by the angular velocity of the ancestors
    virtual const Vec3& getVelocityBasedLinearAccelerationInWorld() const;
    /// Linear acceleration of the origin induced by the angular velocity of the ancestors
    virtual void setVelocityBasedLinearAccelerationInWorld(const Vec3& )
    {}
    /// @}


    /// Gravity in local coordinates
    virtual Vec3 getLocalGravity() const;
    /// Gravity in local coordinates
    //virtual void setGravity( const Vec3& ) { }
    /// Gravity in world coordinates
    virtual const Vec3& getGravityInWorld() const;
    /// Gravity in world coordinates
    virtual void setGravityInWorld( const Vec3& )
    { }

    /// @name Variables
    /// @{

    /// Mechanical Degrees-of-Freedom
    virtual BaseObject* getMechanicalModel() const;

    /// Topology
    virtual BaseObject* getTopology() const;

    /// Topology
    virtual BaseObject* getMainTopology() const;

    /// @}

    /// @name Parameters Setters
    /// @{


    /// Simulation timestep
    virtual void setDt( double /*dt*/ )
    { }

    /// Animation flag
    virtual void setAnimate(bool /*val*/)
    { }

    /// MultiThreading activated
    virtual void setMultiThreadSimulation(bool /*val*/)
    { }

    /// Display flags: Collision Models
    virtual void setShowCollisionModels(bool /*val*/)
    { }

    /// Display flags: Bounding Collision Models
    virtual void setShowBoundingCollisionModels(bool /*val*/)
    { }

    /// Display flags: Behavior Models
    virtual void setShowBehaviorModels(bool /*val*/)
    { }

    /// Display flags: Visual Models
    virtual void setShowVisualModels(bool /*val*/)
    { }

    /// Display flags: Mappings
    virtual void setShowMappings(bool /*val*/)
    { }

    /// Display flags: Mechanical Mappings
    virtual void setShowMechanicalMappings(bool /*val*/)
    { }

    /// Display flags: ForceFields
    virtual void setShowForceFields(bool /*val*/)
    { }

    /// Display flags: InteractionForceFields
    virtual void setShowInteractionForceFields(bool /*val*/)
    { }

    /// Display flags: WireFrame
    virtual void setShowWireFrame(bool /*val*/)
    { }

    /// Display flags: Normals
    virtual void setShowNormals(bool /*val*/)
    { }

    /// Multiresolution
    virtual bool setCurrentLevel(int ) {return false;} ///< set the current level, return false if l >= coarsestLevel
    virtual void setCoarsestLevel(int ) {}
    virtual void setFinestLevel(int ) {}

    /// @}

    /// @name Variables Setters
    /// @{

    /// Mechanical Degrees-of-Freedom
    virtual void setMechanicalModel( Abstract::BaseObject* )
    { }

    /// Topology
    virtual void setTopology( Abstract::BaseObject* )
    { }

    /// @}

    /// @name Adding/Removing objects. Note that these methods can fail if the context don't support attached objects
    /// @{

    /// Add an object, or return false if not supported
    virtual bool addObject( BaseObject* /*obj*/ )
    {
        return false;
    }

    /// Remove an object, or return false if not supported
    virtual bool removeObject( BaseObject* /*obj*/ )
    {
        return false;
    }

    /// @}

    /// @name Actions.
    /// @{

    /// Propagate an event
    virtual void propagateEvent( Event* );

    /// apply an action
    virtual void executeAction( Components::Graph::Action* );

    /// @}



};

} // namespace Abstract

} // namespace Sofa

#endif


