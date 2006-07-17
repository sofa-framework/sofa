#ifndef SOFA_ABSTRACT_BASECONTEXT_H
#define SOFA_ABSTRACT_BASECONTEXT_H

#include "Base.h"

namespace Sofa
{

namespace Abstract
{

class BaseObject;

/// Base class for storing shared variables and parameters.
class BaseContext : public virtual Base
{
public:
    BaseContext();
    virtual ~BaseContext();

    static BaseContext* getDefault();

    /// @name Parameters
    /// @{

    /// Gravity in the local coordinate system as a pointer to 3 doubles
    virtual const double* getGravity() const;

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


    /// @name Local Coordinates System
    /// @{
    // Projection from the local coordinate system to the world coordinate system: translation part.
    /// Returns a pointer to 3 doubles
    //virtual const double* getLocalToWorldTranslation() const;

    /// Projection from the local coordinate system to the world coordinate system: rotation part.
    /// Returns a pointer to a 3x3 matrix (9 doubles, row-major format)
    //virtual const double* getLocalToWorldRotationMatrix() const;

    /// Projection from the local coordinate system to the world coordinate system: rotation part.
    /// Returns a pointer to a quaternion (4 doubles, <x,y,z,w> )
    //virtual const double* getLocalToWorldRotationQuat() const;

    /// Compute the global 4x4 matrix in row-major format
    //void computeLocalToWorldMatrixRowMajor(double* m) const;



    /// Projection from the local coordinate system to the world coordinate system: translation part.
    /// Returns a pointer to 3 doubles
    virtual const double* getLocalToWorldTranslation() const;

    /// Projection from the local coordinate system to the world coordinate system: rotation part.
    /// Returns a pointer to a 3x3 matrix (9 doubles, row-major format)
    virtual const double* getLocalToWorldRotationMatrix() const;

    /// Projection from the local coordinate system to the world coordinate system: rotation part.
    /// Returns a pointer to a quaternion (4 doubles, <x,y,z,w> )
    virtual const double* getLocalToWorldRotationQuat() const;

    /// Compute the global 4x4 matrix in row-major format
    void computeLocalToWorldMatrixRowMajor(double* m) const;

    /// Compute the global 4x4 matrix in column-major (OpenGL) format
    void computeLocalToWorldMatrixColumnMajor(double* m) const;

    /// Velocity of the local frame in the world coordinate system. The linear velocity is expressed at the origin of the world coordinate system.
    /// Returns a pointer to 6 doubles (3 doubles for linear velocity, 3 doubles for angular velocity)
    //virtual const double* getSpatialVelocity() const;


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


    /// @name Variables
    /// @{

    /// Mechanical Degrees-of-Freedom
    virtual BaseObject* getMechanicalModel() const;

    /// Topology
    virtual BaseObject* getTopology() const;

    /// @}

    /// @name Parameters Setters
    /// @{

    /// Gravity in local coordinates
    virtual void setGravity( const double* /*g*/ ) { }

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

    /// Projection from the local frame to the world frame
    virtual void setLocalToWorld( const double* /*translation*/, const double* /*rotationQuat*/, const double* /*rotationMatrix*/) { }

    /// Velocity of the local frame with respect the world coordinate system, expressed in the world coordinate system, at the origin of the world coordinate system
    virtual void setLinearVelocity( const double* ) { }

    /// Velocity of the local frame with respect the world coordinate system, expressed in the world coordinate system, at the origin of the world coordinate system
    virtual void setAngularVelocity( const double* ) { }

    /// Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame
    virtual void setLinearAcceleration( const double* ) { }

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
