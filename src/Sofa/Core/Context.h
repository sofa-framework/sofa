// Author: Fran�is Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef	SOFA_CORE_CONTEXT_H
#define	SOFA_CORE_CONTEXT_H

#include "Sofa/Abstract/BaseContext.h"

#include <iostream>
#include <map>

namespace Sofa
{

namespace Core
{
using Sofa::Components::Common::DataField;

// struct ContextData
// {
// 	typedef Abstract::BaseContext::Frame Frame;
// 	typedef Abstract::BaseContext::Vec3 Vec3;
// 	typedef Abstract::BaseContext::Quat Quat;
// 	typedef Abstract::BaseContext::SpatialVector SpatialVector;
//
//
//         DataField<Vec3> worldGravity_;  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
//         DataField<double> dt_;
//         DataField<double> time_;
//         DataField<bool> animate_;
//         DataField<bool> showCollisionModels_;
//         DataField<bool> showBoundingCollisionModels_;
//         DataField<bool> showBehaviorModels_;
//         DataField<bool> showVisualModels_;
//         DataField<bool> showMappings_;
//         DataField<bool> showMechanicalMappings_;
//         DataField<bool> showForceFields_;
//         DataField<bool> showInteractionForceFields_;
//         DataField<bool> showWireFrame_;
//         DataField<bool> showNormals_;
//         DataField<bool> multiThreadSimulation_;
//
// //         DataField<Vec3> f_gravity;
// //         DataField<double> f_dt;
// //         DataField<double> f_time;
// //         DataField<bool> f_animate;
// //         DataField<bool> f_showCollisionModels;
// //         DataField<bool> f_showBoundingCollisionModels;
// //         DataField<bool> f_showBehaviorModels;
// //         DataField<bool> f_showVisualModels;
// //         DataField<bool> f_showMappings;
// //         DataField<bool> f_showMechanicalMappings;
// //         DataField<bool> f_showForceFields;
// //         DataField<bool> f_showInteractionForceFields;
// //         DataField<bool> f_showWireFrame;
// //         DataField<bool> f_showNormals;
// //         DataField<bool> f_multiThreadSimulation;
//
//
// 	Frame localFrame_;
//         SpatialVector spatialVelocityInWorld_;
// 	Vec3 velocityBasedLinearAccelerationInWorld_;
// 	//double localToWorldTranslation_[3];  ///< Used to project from the local coordinate system to the world coordinate system
// 	//double localToWorldRotationQuat_[4];  ///< Used to project from the local coordinate system to the world coordinate system
// 	//double localToWorldRotationMatrix_[9];  ///< Used to project from the local coordinate system to the world coordinate system
// 	//double linearVelocity_[3]; ///< Velocity in the local frame, defined in the world coordinate system
// 	//double angularVelocity_[3]; ///< Velocity in the local frame, defined in the world coordinate system
// 	//double linearAcceleration_[3]; ///< Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame
// };

class Context : public Abstract::BaseContext
{

public:
    // begin ContextData
    typedef Abstract::BaseContext::Frame Frame;
    typedef Abstract::BaseContext::Vec3 Vec3;
    typedef Abstract::BaseContext::Quat Quat;
    typedef Abstract::BaseContext::SpatialVector SpatialVector;


    DataField<Vec3> worldGravity_;  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
    DataField<double> dt_;
    DataField<double> time_;
    DataField<bool> animate_;
    DataField<bool> showCollisionModels_;
    DataField<bool> showBoundingCollisionModels_;
    DataField<bool> showBehaviorModels_;
    DataField<bool> showVisualModels_;
    DataField<bool> showMappings_;
    DataField<bool> showMechanicalMappings_;
    DataField<bool> showForceFields_;
    DataField<bool> showInteractionForceFields_;
    DataField<bool> showWireFrame_;
    DataField<bool> showNormals_;
    DataField<bool> multiThreadSimulation_;

    Frame localFrame_;
    SpatialVector spatialVelocityInWorld_;
    Vec3 velocityBasedLinearAccelerationInWorld_;

    // end ContextData

//     typedef Abstract::BaseContext::Frame Frame;
// 	typedef Abstract::BaseContext::Vec3 Vec3;
// 	typedef Abstract::BaseContext::Quat Quat;
// 	typedef Abstract::BaseContext::SpatialVector SpatialVector;


    Context();
    virtual ~Context()
    {}
    static BaseContext* getDefault();



    /// @name Parameters
    /// @{

    /// Gravity in the local coordinate system
    virtual Vec3 getLocalGravity() const;
    /// Gravity in the local coordinate system
    //virtual void setGravity(const Vec3& );
    /// Gravity in world coordinates
    virtual const Vec3& getGravityInWorld() const;
    /// Gravity in world coordinates
    virtual void setGravityInWorld( const Vec3& );

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

    /// @}


    /// @name Local Coordinate System
    /// @{
    /// Projection from the local coordinate system to the world coordinate system.
    virtual const Frame& getPositionInWorld() const;
    /// Projection from the local coordinate system to the world coordinate system.
    virtual void setPositionInWorld(const Frame&);

    /// Spatial velocity (linear, angular) of the local frame with respect to the world
    virtual const SpatialVector& getVelocityInWorld() const;
    /// Spatial velocity (linear, angular) of the local frame with respect to the world
    virtual void setVelocityInWorld(const SpatialVector&);

    /// Linear acceleration of the origin induced by the angular velocity of the ancestors
    virtual const Vec3& getVelocityBasedLinearAccelerationInWorld() const;
    /// Linear acceleration of the origin induced by the angular velocity of the ancestors
    virtual void setVelocityBasedLinearAccelerationInWorld(const Vec3& );
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

    /// Display flags: Bounding Collision Models
    virtual void setShowBoundingCollisionModels(bool val);

    /// Display flags: Visual Models
    virtual void setShowVisualModels(bool val);

    /// Display flags: Mappings
    virtual void setShowMappings(bool val);

    /// Display flags: Mechanical Mappings
    virtual void setShowMechanicalMappings(bool val);

    /// Display flags: ForceFields
    virtual void setShowForceFields(bool val);

    /// Display flags: InteractionForceFields
    virtual void setShowInteractionForceFields(bool val);

    /// Display flags: WireFrame
    virtual void setShowWireFrame(bool val);

    /// Display flags: Normals
    virtual void setShowNormals(bool val);


    /// @}

    //static Context getDefault();

    void copyContext(const Context& c);

    friend std::ostream& operator << (std::ostream& out, const Context& c );


};

} // namespace Core

} // namespace Sofa

#endif
