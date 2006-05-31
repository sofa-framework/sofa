#include "BaseContext.h"
#include "BaseObject.h"

namespace Sofa
{

namespace Abstract
{

BaseContext::BaseContext()
{
}

BaseContext::~BaseContext()
{
}

BaseContext* BaseContext::getDefault()
{
    static BaseContext defaultContext;
    return &defaultContext;
}

////////////////
// Parameters //
////////////////

/// Gravity in the local coordinate system as a pointer to 3 doubles
const double* BaseContext::getGravity() const
{
    static const double G[3]= {0,-9.81,0};
    return G;
}

/// Simulation timestep
double BaseContext::getDt() const
{
    return 0.01;
}

/// Animation flag
bool BaseContext::getAnimate() const
{
    return true;
}

/// MultiThreading activated
bool BaseContext::getMultiThreadSimulation() const
{
    return false;
}

/// Display flags: Collision Models
bool BaseContext::getShowCollisionModels() const
{
    return true;
}

/// Display flags: Behavior Models
bool BaseContext::getShowBehaviorModels() const
{
    return true;
}

/// Display flags: Visual Models
bool BaseContext::getShowVisualModels() const
{
    return true;
}

/// Display flags: Mappings
bool BaseContext::getShowMappings() const
{
    return true;
}

/// Display flags: ForceFields
bool BaseContext::getShowForceFields() const
{
    return true;
}


//////////////////////////////
// Local Coordinates System //
//////////////////////////////


/// Projection from the local coordinate system to the world coordinate system: translation part.
/// Returns a pointer to 3 doubles
const double* BaseContext::getLocalToWorldTranslation() const
{
    static const double defaultVal[3]= {0,0,0};
    return defaultVal;
}

/// Projection from the local coordinate system to the world coordinate system: rotation part.
/// Returns a pointer to a 3x3 matrix (9 doubles, row-major format)
const double* BaseContext::getLocalToWorldRotationMatrix() const
{
    static const double defaultVal[9]= {1,0,0, 0,1,0, 0,0,1};
    return defaultVal;
}

/// Projection from the local coordinate system to the world coordinate system: rotation part.
/// Returns a pointer to a quaternion (4 doubles, <x,y,z,w> )
const double* BaseContext::getLocalToWorldRotationQuat() const
{
    static const double defaultVal[4]= {0,0,0,1};
    return defaultVal;
}

/// Compute the global 4x4 matrix in row-major format
void BaseContext::computeLocalToWorldMatrixRowMajor(double* m) const
{
    const double* t = getLocalToWorldTranslation();
    const double* r = getLocalToWorldRotationMatrix();
    m[0*4+0]=r[0*3+0];  m[0*4+1]=r[0*3+1];  m[0*4+2]=r[0*3+2];  m[0*4+3]=t[0];
    m[1*4+0]=r[1*3+0];  m[1*4+1]=r[1*3+1];  m[1*4+2]=r[1*3+2];  m[1*4+3]=t[1];
    m[2*4+0]=r[2*3+0];  m[2*4+1]=r[2*3+1];  m[2*4+2]=r[2*3+2];  m[2*4+3]=t[2];
    m[3*4+0]=0;         m[3*4+1]=0;         m[3*4+2]=0;         m[3*4+3]=1;
}

/// Compute the global 4x4 matrix in column-major (OpenGL) format
void BaseContext::computeLocalToWorldMatrixColumnMajor(double* m) const
{
    const double* t = getLocalToWorldTranslation();
    const double* r = getLocalToWorldRotationMatrix();
    m[0+4*0]=r[0*3+0];  m[0+4*1]=r[0*3+1];  m[0+4*2]=r[0*3+2];  m[0+4*3]=t[0];
    m[1+4*0]=r[1*3+0];  m[1+4*1]=r[1*3+1];  m[1+4*2]=r[1*3+2];  m[1+4*3]=t[1];
    m[2+4*0]=r[2*3+0];  m[2+4*1]=r[2*3+1];  m[2+4*2]=r[2*3+2];  m[2+4*3]=t[2];
    m[3+4*0]=0;         m[3+4*1]=0;         m[3+4*2]=0;         m[3+4*3]=1;
}
/*
/// Velocity of the local frame in the world coordinate system. The linear velocity is expressed at the origin of the world coordinate system.
/// Returns a pointer to 6 doubles (3 doubles for linear velocity, 3 doubles for angular velocity)
const double* BaseContext::getSpatialVelocity() const
{
	static const double defaultVal[6]={0,0,0,0,0,0};
	return defaultVal;
}
*/

/// Velocity of the local frame in the world coordinate system. The linear velocity is expressed at the origin of the world coordinate system.
/// Returns a pointer to 3 doubles
const double* BaseContext::getLinearVelocity() const
{
    static const double defaultVal[3]= {0,0,0};
    return defaultVal;
}

/// Velocity of the local frame in the world coordinate system.
/// Returns a pointer to 3 doubles
const double* BaseContext::getAngularVelocity() const
{
    static const double defaultVal[3]= {0,0,0};
    return defaultVal;
}

/// Acceleration of the origin of the frame due to the velocities of the ancestors of the current frame.
/// Returns a pointer to 3 doubles
const double* BaseContext::getLinearAcceleration() const
{
    static const double defaultVal[3]= {0,0,0};
    return defaultVal;
}


///////////////
// Variables //
///////////////


/// Mechanical Degrees-of-Freedom
BaseObject* BaseContext::getMechanicalModel() const
{
    return NULL;
}

/// Topology
BaseObject* BaseContext::getTopology() const
{
    return NULL;
}

} // namespace Abstract

} // namespace Sofa
