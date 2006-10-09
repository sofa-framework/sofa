#include "Context.h"
#include "Sofa/Components/Graph/MechanicalAction.h"
#include "Sofa/Components/Graph/MechanicalVPrintAction.h"

namespace Sofa
{
namespace Core
{

Context::Context()
    : result(0)
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

Abstract::BaseContext* Context::getDefault()
{
    static Context defaultContext;
    return &defaultContext;
}


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
    return dt_;
}

/// Simulation time
double Context::getTime() const
{
    return time_;
}

/// Gravity vector in local coordinates
// const Context::Vec3& Context::getGravity() const
// {
// 	return gravity_;
// }

/// Gravity vector in local coordinates
Context::Vec3 Context::getLocalGravity() const
{
    return getPositionInWorld().backProjectVector(worldGravity_);
}

/// Gravity vector in world coordinates
const Context::Vec3& Context::getGravityInWorld() const
{
    return worldGravity_;
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

/// Display flags: Bounding Collision Models
bool Context::getShowBoundingCollisionModels() const
{
    return showBoundingCollisionModels_;
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

/// Display flags: Mechanical Mappings
bool Context::getShowMechanicalMappings() const
{
    return showMechanicalMappings_;
}

/// Display flags: ForceFields
bool Context::getShowForceFields() const
{
    return showForceFields_;
}

/// Display flags: InteractionForceFields
bool Context::getShowInteractionForceFields() const
{
    return showInteractionForceFields_;
}

/// Display flags: WireFrame
bool Context::getShowWireFrame() const
{
    return showWireFrame_;
}

/// Display flags: Normal
bool Context::getShowNormals() const
{
    return showNormals_;
}

//===============================================================================

/// Simulation timestep
void Context::setDt(double val)
{
    dt_ = val;
}

/// Simulation time
void Context::setTime(double val)
{
    time_ = val;
}

/// Gravity vector
// void Context::setGravity(const Vec3& g)
// {
// 	gravity_ = g;
// }

/// Gravity vector
void Context::setGravityInWorld(const Vec3& g)
{
    worldGravity_ = g;
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

/// Display flags: Bounding Collision Models
void Context::setShowBoundingCollisionModels(bool val)
{
    showBoundingCollisionModels_ = val;
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

/// Display flags: Mechanical Mappings
void Context::setShowMechanicalMappings(bool val)
{
    showMechanicalMappings_ = val;
}

/// Display flags: ForceFields
void Context::setShowForceFields(bool val)
{
    showForceFields_ = val;
}

/// Display flags: InteractionForceFields
void Context::setShowInteractionForceFields(bool val)
{
    showInteractionForceFields_ = val;
}

/// Display flags: WireFrame
void Context::setShowWireFrame(bool val)
{
    showWireFrame_ = val;
}

/// Display flags: Normals
void Context::setShowNormals(bool val)
{
    showNormals_ = val;
}



void Context::copyContext(const Context& c)
{
    *static_cast<ContextData*>(this) = *static_cast<const ContextData *>(&c);
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

using namespace Core::Encoding;
using namespace Components::Graph;

Context::VectorIndexAlloc::VectorIndexAlloc()
    : maxIndex(V_FIRST_DYNAMIC_INDEX-1)
{}

unsigned int Context::VectorIndexAlloc::alloc()
{
    int v;
    if (vfree.empty())
        v = ++maxIndex;
    else
    {
        v = *vfree.begin();
        vfree.erase(vfree.begin());
    }
    vused.insert(v);
    return v;
}








bool Context::VectorIndexAlloc::free(unsigned int v)
{
    if (v < V_FIRST_DYNAMIC_INDEX)
        return false;
    // @TODO: Check for errors
    vused.erase(v);
    vfree.insert(v);
    return true;
}

//                 Context::Context(GNode* node)
//                 : node(node), result(0)
//                 {}

/// Wait for the completion of previous operations and return the result of the last v_dot call
double Context::finish()
{
    return result;
}

VecId Context::v_alloc(VecType t)
{
    VecId v(t, vectors[t].alloc());
    MechanicalVAllocAction(v).execute( this );
    return v;
}

void Context::v_free(VecId v)
{
    if (vectors[v.type].free(v.index))
        MechanicalVFreeAction(v).execute( this );
}

void Context::v_clear(VecId v) ///< v=0
{
    MechanicalVOpAction(v).execute( this );
}

void Context::v_eq(VecId v, VecId a) ///< v=a
{
    MechanicalVOpAction(v,a).execute( this );
}

void Context::v_peq(VecId v, VecId a, double f) ///< v+=f*a
{
    MechanicalVOpAction(v,v,a,f).execute( this );
}
void Context::v_teq(VecId v, double f) ///< v*=f
{
    MechanicalVOpAction(v,VecId::null(),v,f).execute( this );
}
void Context::v_dot(VecId a, VecId b) ///< a dot b ( get result using finish )
{
    result = 0;
    MechanicalVDotAction(a,b,&result).execute( this );
}

void Context::propagateDx(VecId dx)
{
    MechanicalPropagateDxAction(dx).execute( this );
}

void Context::projectResponse(VecId dx)
{
    MechanicalApplyConstraintsAction(dx).execute( this );
}

void Context::addMdx(VecId res, VecId dx)
{
    MechanicalAddMDxAction(res,dx).execute( this );
}

void Context::integrateVelocity(VecId res, VecId x, VecId v, double dt)
{
    MechanicalVOpAction(res,x,v,dt).execute( this );
}

void Context::accFromF(VecId a, VecId f)
{
    MechanicalAccFromFAction(a,f).execute( this );
}

void Context::propagatePositionAndVelocity(double t, VecId x, VecId v)
{
    MechanicalPropagatePositionAndVelocityAction(t,x,v).execute( this );
}

void Context::computeForce(VecId result)
{
    MechanicalResetForceAction(result).execute( this );
    finish();
    MechanicalComputeForceAction(result).execute( this );
}

void Context::computeDf(VecId df)
{
    MechanicalResetForceAction(df).execute( this );
    finish();
    MechanicalComputeDfAction(df).execute( this );
}

void Context::computeAcc(double t, VecId a, VecId x, VecId v)
{
    VecId f = VecId::force();
    propagatePositionAndVelocity(t, x, v);
    computeForce(f);
    accFromF(a, f);
    projectResponse(a);
}

void Context::print( VecId v, std::ostream& out )
{
    MechanicalVPrintAction(v,out).execute( this );
}

//                 double Context::getTime() const
// {
//     return this->getTime();
// }

// Matrix Computing in ForceField

void Context::getMatrixDimension(unsigned int * const nbRow, unsigned int * const nbCol)
{
    MechanicalGetMatrixDimensionAction(nbRow, nbCol).execute( this );
}


void Context::computeMatrix(Components::Common::SofaBaseMatrix *mat, double mFact, double bFact, double kFact, unsigned int offset)
{
    if (mat != NULL)
    {
        MechanicalComputeMatrixAction(mat, mFact, bFact, kFact, offset).execute( this );
    }
}


void Context::computeOpVector(Components::Common::SofaBaseVector *vect, unsigned int offset)
{
    if (vect != NULL)
    {
        MechanicalComputeVectorAction(vect, offset).execute( this );
    }
}


void Context::matResUpdatePosition(Components::Common::SofaBaseVector *vect, unsigned int offset)
{
    if (vect != NULL)
    {
        MechanicalMatResUpdatePositionAction(vect, offset).execute( this );
    }
}




} // namespace Core

} // namespace Sofa


