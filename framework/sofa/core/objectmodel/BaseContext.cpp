/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/Shader.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{

namespace core
{

namespace objectmodel
{

BaseContext::BaseContext()
{}

BaseContext::~BaseContext()
{}

BaseContext* BaseContext::getDefault()
{
    static BaseContext defaultContext;
    return &defaultContext;
}

////////////////
// Parameters //
////////////////

/// The Context is active
bool BaseContext::isActive() const { return true;};
#ifdef SOFA_SMP
bool BaseContext::is_partition() const { return false;};
#endif

/// Gravity in the local coordinate system
BaseContext::Vec3 BaseContext::getLocalGravity() const
{
    static const Vec3 G((SReal)0,(SReal)-9.81,(SReal)0);
    return G;
}

/// Gravity in the world coordinate system
const BaseContext::Vec3& BaseContext::getGravityInWorld() const
{
    static const Vec3 G((SReal)0,(SReal)-9.81,(SReal)0);
    return G;
}

/// Simulation timestep
double BaseContext::getDt() const
{
    return 0.01;
}

/// Simulation time
double BaseContext::getTime() const
{
    return 0.0;
}

/// Animation flag
bool BaseContext::getAnimate() const
{
    return true;
}

#ifdef SOFA_SMP
int BaseContext::getProcessor() const
{
    return -1;
}
Iterative::IterativePartition* BaseContext::getPartition() const
{
    return 0;
}
#endif

/// Display flags: Collision Models
bool BaseContext::getShowCollisionModels() const
{
    return false;
}

/// Display flags: Bounding Collision Models
bool BaseContext::getShowBoundingCollisionModels() const
{
    return false;
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
    return false;
}

/// Display flags: Mechanical Mappings
bool BaseContext::getShowMechanicalMappings() const
{
    return false;
}

/// Display flags: ForceFields
bool BaseContext::getShowForceFields() const
{
    return false;
}

/// Display flags: InteractionForceFields
bool BaseContext::getShowInteractionForceFields() const
{
    return false;
}

/// Display flags: WireFrame
bool BaseContext::getShowWireFrame() const
{
    return false;
}

/// Display flags: Normals
bool BaseContext::getShowNormals() const
{
    return false;
}

#ifdef SOFA_SMP
bool BaseContext::getShowProcessorColor() const
{
    return false;
}
#endif

#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MULTIRESOLUTION
/// Multiresolution
int BaseContext::getCurrentLevel() const
{
    return 0;
}
int BaseContext::getCoarsestLevel() const
{
    return 0;
}
int BaseContext::getFinestLevel() const
{
    return 0;
}
// unsigned int BaseContext::nbLevels() const
// {
// 	return getCoarsestLevel() - getFinestLevel() + 1;
// }
#endif
#endif // SOFA_DEV

//////////////////////////////
// Local Coordinates System //
//////////////////////////////


/// Projection from the local coordinate system to the world coordinate system.
const BaseContext::Frame& BaseContext::getPositionInWorld() const
{
    static const Frame f;
    return f;
}

/// Spatial velocity (linear, angular) of the local frame with respect to the world
const BaseContext::SpatialVector& BaseContext::getVelocityInWorld() const
{
    static const SpatialVector v( Vec3(0,0,0), Vec3(0,0,0) );
    return v;
}

/// Linear acceleration of the origin induced by the angular velocity of the ancestors
const BaseContext::Vec3& BaseContext::getVelocityBasedLinearAccelerationInWorld() const
{
    static const Vec3 a(0,0,0);
    return a;
}


////////////////
// Containers //
////////////////

/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BaseContext::getObject(const ClassInfo& /*class_info*/, SearchDirection /*dir*/) const
{
    std::cerr << "BaseContext: calling unimplemented getObject method" << std::endl;
    return NULL;
}

/// Generic object access, given a set of required tags, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BaseContext::getObject(const ClassInfo& /*class_info*/, const TagSet& /*tags*/, SearchDirection /*dir*/) const
{
    std::cerr << "BaseContext: calling unimplemented getObject method" << std::endl;
    return NULL;
}

/// Generic object access, given a path from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BaseContext::getObject(const ClassInfo& /*class_info*/, const std::string& /*path*/) const
{
    std::cerr << "BaseContext: calling unimplemented getObject method" << std::endl;
    return NULL;
}

/// Generic list of objects access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void BaseContext::getObjects(const ClassInfo& /*class_info*/, GetObjectsCallBack& /*container*/, SearchDirection /*dir*/) const
{
    std::cerr << "BaseContext: calling unimplemented getObjects method" << std::endl;
}

/// Generic list of objects access, given a set of required tags, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void BaseContext::getObjects(const ClassInfo& /*class_info*/, GetObjectsCallBack& /*container*/, const TagSet& /*tags*/, SearchDirection /*dir*/) const
{
    std::cerr << "BaseContext: calling unimplemented getObject method" << std::endl;
}

/// Degrees-of-Freedom
BaseObject* BaseContext::getState() const
{
    return this->get<sofa::core::BaseState>();
}

/// Mechanical Degrees-of-Freedom
BaseObject* BaseContext::getMechanicalState() const
{
    return this->get<sofa::core::behavior::BaseMechanicalState>();
}

/// Mass
BaseObject* BaseContext::getMass() const
{
    return this->get<sofa::core::behavior::BaseMass>();
}


/// Topology
core::topology::Topology* BaseContext::getTopology() const
{
    return this->get<sofa::core::topology::Topology>();
}
/// Mesh Topology (unified interface for both static and dynamic topologies)
core::topology::BaseMeshTopology* BaseContext::getMeshTopology() const
{
    return this->get<sofa::core::topology::BaseMeshTopology>();
}

/// Shader
BaseObject* BaseContext::getShader() const
{
    return this->get<sofa::core::Shader>();
    return NULL;
}

/// Propagate an event
void BaseContext::propagateEvent( const core::ExecParams* /* PARAMS FIRST */, Event* )
{
    serr<<"WARNING !!! BaseContext::propagateEvent not overloaded, does nothing"<<sendl;
}

void BaseContext::executeVisitor( simulation::Visitor* )
{
    serr<<"WARNING !!! BaseContext::executeVisitor not overloaded, does nothing"<<sendl;
    //assert(false);
}

std::ostream& operator << (std::ostream& out, const BaseContext& c )
{
    out<<std::endl<<"local gravity = "<<c.getLocalGravity();
    out<<std::endl<<"transform from local to world = "<<c.getPositionInWorld();
    //out<<std::endl<<"transform from world to local = "<<c.getWorldToLocal();
    out<<std::endl<<"spatial velocity = "<<c.getVelocityInWorld();
    out<<std::endl<<"acceleration of the origin = "<<c.getVelocityBasedLinearAccelerationInWorld();
    out<<std::endl<<"showBehaviorModels = "<<c.getShowBehaviorModels();
    return out;
}




} // namespace objectmodel

} // namespace core

} // namespace sofa

