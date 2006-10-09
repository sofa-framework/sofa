#ifndef SOFA_ABSTRACT_BASECONTEXT_H
#define SOFA_ABSTRACT_BASECONTEXT_H

#include "Base.h"
#include <Sofa/Components/Common/SolidTypes.h>
#include <Sofa/Components/Common/SofaBaseMatrix.h>
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

    //static BaseContext* getDefault();

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

    /// apply an action
    virtual void executeAction( Components::Graph::Action* );

    /// @}


public:
    //MechanicalIntegration(GNode* node);

    //virtual double getTime() const=0;

    /// Wait for the completion of previous operations and return the result of the last v_dot call
    virtual double finish()=0;

    virtual VecId v_alloc(Core::Encoding::VecType t)=0;
    virtual void v_free(VecId v)=0;

    virtual void v_clear(VecId v)=0; ///< v=0
    virtual void v_eq(VecId v, VecId a)=0; ///< v=a
    virtual void v_peq(VecId v, VecId a, double f=1.0)=0; ///< v+=f*a
    virtual void v_teq(VecId v, double f)=0; ///< v*=f
    virtual void v_dot(VecId a, VecId b)=0; ///< a dot b ( get result using finish )
    virtual void propagateDx(VecId dx)=0;
    virtual void projectResponse(VecId dx)=0;
    virtual void addMdx(VecId res, VecId dx)=0;
    virtual void integrateVelocity(VecId res, VecId x, VecId v, double dt)=0;
    virtual void accFromF(VecId a, VecId f)=0;
    virtual void propagatePositionAndVelocity(double t, VecId x, VecId v)=0;

    virtual void computeForce(VecId result)=0;
    virtual void computeDf(VecId df)=0;
    virtual void computeAcc(double t, VecId a, VecId x, VecId v)=0;

    virtual void computeMatrix(Components::Common::SofaBaseMatrix *mat=NULL, double mFact=1.0, double bFact=1.0, double kFact=1.0, unsigned int offset=0)=0;
    virtual void getMatrixDimension(unsigned int * const, unsigned int * const)=0;
    virtual void computeOpVector(Components::Common::SofaBaseVector *vect=NULL, unsigned int offset=0)=0;
    virtual void matResUpdatePosition(Components::Common::SofaBaseVector *vect=NULL, unsigned int offset=0)=0;

    virtual void print( VecId v, std::ostream& out )=0;



};

} // namespace Abstract

} // namespace Sofa

#endif


