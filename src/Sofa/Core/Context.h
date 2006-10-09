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

struct ContextData
{
    typedef Abstract::BaseContext::Frame Frame;
    typedef Abstract::BaseContext::Vec3 Vec3;
    typedef Abstract::BaseContext::Quat Quat;
    typedef Abstract::BaseContext::SpatialVector SpatialVector;

    //double gravity_[3];  ///< Gravity
    //double worldGravity_[3];  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
    //Vec3 gravity_;  ///< Gravity
    Vec3 worldGravity_;  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
    double dt_;
    double time_;
    bool animate_;
    bool showCollisionModels_;
    bool showBoundingCollisionModels_;
    bool showBehaviorModels_;
    bool showVisualModels_;
    bool showMappings_;
    bool showMechanicalMappings_;
    bool showForceFields_;
    bool showInteractionForceFields_;
    bool showWireFrame_;
    bool showNormals_;
    bool multiThreadSimulation_;


    Frame localFrame_;
    SpatialVector spatialVelocityInWorld_;
    Vec3 velocityBasedLinearAccelerationInWorld_;
    //double localToWorldTranslation_[3];  ///< Used to project from the local coordinate system to the world coordinate system
    //double localToWorldRotationQuat_[4];  ///< Used to project from the local coordinate system to the world coordinate system
    //double localToWorldRotationMatrix_[9];  ///< Used to project from the local coordinate system to the world coordinate system
    //double linearVelocity_[3]; ///< Velocity in the local frame, defined in the world coordinate system
    //double angularVelocity_[3]; ///< Velocity in the local frame, defined in the world coordinate system
    //double linearAcceleration_[3]; ///< Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame
};

class Context : public Abstract::BaseContext, private ContextData
{

public:
    typedef Abstract::BaseContext::Frame Frame;
    typedef Abstract::BaseContext::Vec3 Vec3;
    typedef Abstract::BaseContext::Quat Quat;
    typedef Abstract::BaseContext::SpatialVector SpatialVector;


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

protected:

    class VectorIndexAlloc
    {
    protected:
        std::set<unsigned int> vused; ///< Currently in-use vectors
        std::set<unsigned int> vfree; ///< Once used vectors
        unsigned int  maxIndex; ///< Max index used
    public:
        VectorIndexAlloc();
        unsigned int alloc();
        bool free(unsigned int v);
    };
    std::map<Core::Encoding::VecType, VectorIndexAlloc > vectors; ///< Current vectors

    //GNode* node; ///< The root node of the scenegraph
    double result;
public:
    //MechanicalIntegration(GNode* node);

    //double getTime() const;

    /// Wait for the completion of previous operations and return the result of the last v_dot call
    virtual double finish();

    virtual VecId v_alloc(Core::Encoding::VecType t);
    virtual void v_free(VecId v);

    virtual void v_clear(VecId v); ///< v=0
    virtual void v_eq(VecId v, VecId a); ///< v=a
    virtual void v_peq(VecId v, VecId a, double f=1.0); ///< v+=f*a
    virtual void v_teq(VecId v, double f); ///< v*=f
    virtual void v_dot(VecId a, VecId b); ///< a dot b ( get result using finish )
    virtual void propagateDx(VecId dx);
    virtual void projectResponse(VecId dx);
    virtual void addMdx(VecId res, VecId dx);
    virtual void integrateVelocity(VecId res, VecId x, VecId v, double dt);
    virtual void accFromF(VecId a, VecId f);
    virtual void propagatePositionAndVelocity(double t, VecId x, VecId v);

    virtual void computeForce(VecId result);
    virtual void computeDf(VecId df);
    virtual void computeAcc(double t, VecId a, VecId x, VecId v);

    virtual void computeMatrix(Components::Common::SofaBaseMatrix *mat=NULL, double mFact=1.0, double bFact=1.0, double kFact=1.0, unsigned int offset=0);
    virtual void getMatrixDimension(unsigned int * const, unsigned int * const);
    virtual void computeOpVector(Components::Common::SofaBaseVector *vect=NULL, unsigned int offset=0);
    virtual void matResUpdatePosition(Components::Common::SofaBaseVector *vect=NULL, unsigned int offset=0);

    virtual void print( VecId v, std::ostream& out );

};

} // namespace Core

} // namespace Sofa

#endif
