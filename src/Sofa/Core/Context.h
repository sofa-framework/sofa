// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution

#ifndef	SOFA_CORE_Context_h
#define	SOFA_CORE_Context_h

#include <iostream>
using std::endl;
#include "Sofa/Components/Common/Vec.h"
#include <Sofa/Components/Common/SolidTypes.h>


namespace Sofa
{

namespace Core
{

class Context
{

public:
    typedef Components::Common::SolidTypes<float>::Vec Vec;
    typedef Components::Common::SolidTypes<float>::Coord Frame;
    typedef Components::Common::SolidTypes<float>::Deriv SpatialVelocity;
    Context();
    virtual ~Context()
    {}


    /// @name Getters
    /// @{
    /// Gravity in the local coordinate system
    virtual const Vec getGravity() const;

    /// Projection from the local coordinate system to the world coordinate system
    virtual const Frame& getLocalToWorld() const;

    /** Velocity of the local frame in the world coordinate system. The linear velocity is expressed at the origin of the world coordinate system. */
    virtual const SpatialVelocity& getSpatialVelocity() const;

    /// Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame
    virtual const Vec& getLinearAcceleration() const;

    /** Velocity of the local frame in the world coordinate system. The linear velocity is expressed at the origin of the world coordinate system. */
    virtual Vec getLinearVelocity() const;

    /** Velocity of the local frame in the world coordinate system.*/
    virtual Vec getAngularVelocity() const;

    /// Simulation timestep
    virtual float getDt() const;

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

    /// @}

    /// @name Global Parameters Setters
    /// @{

    /// Gravity in local coordinates
    virtual void setGravity( const Vec& g );

    /// Projection from the local frame to the world frame
    virtual void setLocalToWorld( const Frame& f );

    /** Velocity of the local frame with respect the world coordinate system, expressed in the world coordinate system, at the origin of the world coordinate system */
    virtual void setSpatialVelocity( const SpatialVelocity& );

    /// Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame
    virtual void setLinearAcceleration( const Vec& );

    /// Simulation timestep
    virtual void setDt( float dt );

    /// Animation flag
    virtual void setAnimate(bool val);


    /// MultiThreading activated
    virtual void setMultiThreadSimulation(bool val);

    /// Display flags: Collision Models
    virtual void setShowCollisionModels(bool val);

    /// Display flags: Behavior Models
    virtual void setShowBehaviorModels(bool val);

    /// Display flags: Visual Models
    virtual void setShowVisualModels(bool val);

    /// Display flags: Mappings
    virtual void setShowMappings(bool val);

    /// Display flags: ForceFields
    virtual void setShowForceFields(bool val);

    /// @}

    static Context getDefault();

    friend std::ostream& operator << (std::ostream& out, const Context& c );

private:
    Vec gravity_;  ///< Gravity IN THE WORLD COORDINATE SYSTEM. Method getGravity() performs the projection transparently.
    Frame localToWorld_;  ///< Used to project from the local coordinate system to the world coordinate system
    SpatialVelocity spatialVelocity_; ///< Velocity in the local frame, defined in the world coordinate system
    Vec originAcceleration_; ///< Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame
    float dt_;
    bool animate_;
    bool showCollisionModels_;
    bool showBehaviorModels_;
    bool showVisualModels_;
    bool showMappings_;
    bool showForceFields_;
    bool multiThreadSimulation_;
};

}//Core
}//Sofa

#endif

