// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef	SOFA_CORE_CONTEXT_H
#define	SOFA_CORE_CONTEXT_H

#include "Sofa/Abstract/BaseContext.h"

#include <iostream>

namespace Sofa
{

namespace Core
{

struct ContextData
{
    double gravity_[3];  ///< Gravity
    double worldGravity_[3];  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
    double dt_;
    double time_;
    bool animate_;
    bool showCollisionModels_;
    bool showBehaviorModels_;
    bool showVisualModels_;
    bool showMappings_;
    bool showForceFields_;
    bool showWireFrame_;
    bool showNormals_;
    bool multiThreadSimulation_;

    double localToWorldTranslation_[3];  ///< Used to project from the local coordinate system to the world coordinate system
    double localToWorldRotationQuat_[4];  ///< Used to project from the local coordinate system to the world coordinate system
    double localToWorldRotationMatrix_[9];  ///< Used to project from the local coordinate system to the world coordinate system
    double linearVelocity_[3]; ///< Velocity in the local frame, defined in the world coordinate system
    double angularVelocity_[3]; ///< Velocity in the local frame, defined in the world coordinate system
    double linearAcceleration_[3]; ///< Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame
};

class Context : public Abstract::BaseContext, private ContextData
{

public:
    Context();
    virtual ~Context()
    {}


    /// @name Parameters
    /// @{

    /// Gravity in the local coordinate system as a pointer to 3 doubles
    virtual const double* getGravity() const;

    /// Simulation timestep
    virtual double getDt() const;

    /// Simulation time
    virtual double getTime() const;

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


    /// @name Local Coordinates System
    /// @{

    /// Projection from the local coordinate system to the world coordinate system: translation part.
    /// Returns a pointer to 3 doubles
    virtual const double* getLocalToWorldTranslation() const;

    /// Projection from the local coordinate system to the world coordinate system: rotation part.
    /// Returns a pointer to a 3x3 matrix (9 doubles, row-major format)
    virtual const double* getLocalToWorldRotationMatrix() const;

    /// Projection from the local coordinate system to the world coordinate system: rotation part.
    /// Returns a pointer to a quaternion (4 doubles, <x,y,z,w> )
    virtual const double* getLocalToWorldRotationQuat() const;

    /*
    	/// Velocity of the local frame in the world coordinate system. The linear velocity is expressed at the origin of the world coordinate system.
    	/// Returns a pointer to 6 doubles (3 doubles for linear velocity, 3 doubles for angular velocity)
    	virtual const double* getSpatialVelocity() const;
    */

    /// Velocity of the local frame in the world coordinate system. The linear velocity is expressed at the origin of the world coordinate system.
    /// Returns a pointer to 3 doubles
    virtual const double* getLinearVelocity() const;

    /// Velocity of the local frame in the world coordinate system.
    /// Returns a pointer to 3 doubles
    virtual const double* getAngularVelocity() const;

    /// Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame.
    /// Returns a pointer to 3 doubles
    virtual const double* getLinearAcceleration() const;

    /// @}

    /*
        /// @name Variables
        /// @{

    	/// Mechanical Degrees-of-Freedom
    	virtual Abstract::BaseObject* getMechanicalModel() const;

    	/// Topology
    	virtual Abstract::BaseObject* getTopology() const;

    	/// @}
    */

    /// @name Parameters Setters
    /// @{

    /// Gravity in local coordinates
    virtual void setGravity( const double* g );

    /// Simulation timestep
    virtual void setDt( double dt );

    /// Simulation time
    virtual void setTime( double t );

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

    /// Display flags: WireFrame
    virtual void setShowWireFrame(bool val);

    /// Display flags: Normals
    virtual void setShowNormals(bool val);

    /// Projection from the local frame to the world frame
    virtual void setLocalToWorld( const double* translation, const double* rotationQuat, const double* rotationMatrix );

    /// Velocity of the local frame with respect the world coordinate system, expressed in the world coordinate system, at the origin of the world coordinate system
    virtual void setLinearVelocity( const double* );

    /// Velocity of the local frame with respect the world coordinate system, expressed in the world coordinate system, at the origin of the world coordinate system
    virtual void setAngularVelocity( const double* );

    /// Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame
    virtual void setLinearAcceleration( const double* );

    /// @}

    //static Context getDefault();

    void copyContext(const Context& c);

    friend std::ostream& operator << (std::ostream& out, const Context& c );

private:
};

} // namespace Core

} // namespace Sofa

#endif
