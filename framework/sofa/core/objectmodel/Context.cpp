/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/core/objectmodel/Context.h>
// #include <sofa/simulation/tree/Action.h>
// #include <sofa/simulation/tree/Action.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

Context::Context()
    : worldGravity_(dataField(&worldGravity_, Vec3(0,0,0),"gravity","Gravity in the world coordinate system"))
    , dt_(dataField(&dt_,0.01,"dt","Time step"))
    , time_(dataField(&time_,0.,"time","Current time"))
    , animate_(dataField(&animate_,false,"animate","???"))
    , showCollisionModels_(dataField(&showCollisionModels_,false,"showCollisionModels","display flag"))
    , showBoundingCollisionModels_( dataField(&showBoundingCollisionModels_,false,"showBoundingCollisionModels","display flag"))
    , showBehaviorModels_(dataField(&showBehaviorModels_,false,"showBehaviorModels","display flag"))
    , showVisualModels_(dataField(&showVisualModels_,false,"showVisualModels","display flag"))
    , showMappings_(dataField(&showMappings_,false,"showMappings","display flag"))
    , showMechanicalMappings_(dataField(&showMechanicalMappings_,false,"showMechanicalMappings","display flag"))
    , showForceFields_(dataField(&showForceFields_,false,"showForceFields","display flag"))
    , showInteractionForceFields_(dataField(&showInteractionForceFields_,false,"showInteractionForceFields","display flag"))
    , showWireFrame_(dataField(&showWireFrame_,false,"showWireFrame","display flag"))
    , showNormals_(dataField(&showNormals_,false,"showNormals","display flag"))
    , multiThreadSimulation_(dataField(&multiThreadSimulation_,false,"multiThreadSimulation","Apply multithreaded simulation"))
    , currentLevel_(dataField(&currentLevel_,0,"currentLevel","Current level of details"))
    , coarsestLevel_(dataField(&coarsestLevel_,3,"coarsestLevel","Coarsest level of details"))
    , finestLevel_(dataField(&finestLevel_,0,"finestLevel","Finest level of details"))
{
    setPositionInWorld(objectmodel::BaseContext::getPositionInWorld());
    setGravityInWorld(objectmodel::BaseContext::getLocalGravity());
    setVelocityInWorld(objectmodel::BaseContext::getVelocityInWorld());
    setVelocityBasedLinearAccelerationInWorld(objectmodel::BaseContext::getVelocityBasedLinearAccelerationInWorld());
    setDt(objectmodel::BaseContext::getDt());
    setTime(objectmodel::BaseContext::getTime());
    setAnimate(objectmodel::BaseContext::getAnimate());
    setShowCollisionModels(objectmodel::BaseContext::getShowCollisionModels());
    setShowBoundingCollisionModels(objectmodel::BaseContext::getShowBoundingCollisionModels());
    setShowBehaviorModels(objectmodel::BaseContext::getShowBehaviorModels());
    setShowVisualModels(objectmodel::BaseContext::getShowVisualModels());
    setShowMappings(objectmodel::BaseContext::getShowMappings());
    setShowMechanicalMappings(objectmodel::BaseContext::getShowMechanicalMappings());
    setShowForceFields(objectmodel::BaseContext::getShowForceFields());
    setShowInteractionForceFields(objectmodel::BaseContext::getShowInteractionForceFields());
    setShowWireFrame(objectmodel::BaseContext::getShowWireFrame());
    setShowNormals(objectmodel::BaseContext::getShowNormals());
    setMultiThreadSimulation(objectmodel::BaseContext::getMultiThreadSimulation());
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
    return dt_.getValue();
}

/// Simulation time
double Context::getTime() const
{
    return time_.getValue();
}

/// Gravity vector in local coordinates
// const Context::Vec3& Context::getGravity() const
// {
// 	return gravity_;
// }

/// Gravity vector in local coordinates
Context::Vec3 Context::getLocalGravity() const
{
    return getPositionInWorld().backProjectVector(worldGravity_.getValue());
}

/// Gravity vector in world coordinates
const Context::Vec3& Context::getGravityInWorld() const
{
    return worldGravity_.getValue();
}



/// Animation flag
bool Context::getAnimate() const
{
    return animate_.getValue();
}

/// MultiThreading activated
bool Context::getMultiThreadSimulation() const
{
    return multiThreadSimulation_.getValue();
}

/// Display flags: Collision Models
bool Context::getShowCollisionModels() const
{
    return showCollisionModels_.getValue();
}

/// Display flags: Bounding Collision Models
bool Context::getShowBoundingCollisionModels() const
{
    return showBoundingCollisionModels_.getValue();
}

/// Display flags: Behavior Models
bool Context::getShowBehaviorModels() const
{
    return showBehaviorModels_.getValue();
}

/// Display flags: Visual Models
bool Context::getShowVisualModels() const
{
    return showVisualModels_.getValue();
}

/// Display flags: Mappings
bool Context::getShowMappings() const
{
    return showMappings_.getValue();
}

/// Display flags: Mechanical Mappings
bool Context::getShowMechanicalMappings() const
{
    return showMechanicalMappings_.getValue();
}

/// Display flags: ForceFields
bool Context::getShowForceFields() const
{
    return showForceFields_.getValue();
}

/// Display flags: InteractionForceFields
bool Context::getShowInteractionForceFields() const
{
    return showInteractionForceFields_.getValue();
}

/// Display flags: WireFrame
bool Context::getShowWireFrame() const
{
    return showWireFrame_.getValue();
}

/// Display flags: Normal
bool Context::getShowNormals() const
{
    return showNormals_.getValue();
}


// Multiresolution

int Context::getCurrentLevel() const
{
    return currentLevel_.getValue();
}
int Context::getCoarsestLevel() const
{
    return coarsestLevel_.getValue();
}
int Context::getFinestLevel() const
{
    return finestLevel_.getValue();
}


//===============================================================================

/// Simulation timestep
void Context::setDt(double val)
{
    dt_.setValue(val);
}

/// Simulation time
void Context::setTime(double val)
{
    time_.setValue(val);
}

/// Gravity vector
// void Context::setGravity(const Vec3& g)
// {
// 	gravity_ = g;
// }

/// Gravity vector
void Context::setGravityInWorld(const Vec3& g)
{
    worldGravity_ .setValue(g);
}

/// Animation flag
void Context::setAnimate(bool val)
{
    animate_.setValue(val);
}

/// MultiThreading activated
void Context::setMultiThreadSimulation(bool val)
{
    multiThreadSimulation_.setValue(val);
}

/// Display flags: Collision Models
void Context::setShowCollisionModels(bool val)
{
    showCollisionModels_.setValue(val);
}

/// Display flags: Bounding Collision Models
void Context::setShowBoundingCollisionModels(bool val)
{
    showBoundingCollisionModels_.setValue(val);
}

/// Display flags: Behavior Models
void Context::setShowBehaviorModels(bool val)
{
    showBehaviorModels_.setValue(val);
}

/// Display flags: Visual Models
void Context::setShowVisualModels(bool val)
{
    showVisualModels_.setValue(val);
}

/// Display flags: Mappings
void Context::setShowMappings(bool val)
{
    showMappings_.setValue(val);
}

/// Display flags: Mechanical Mappings
void Context::setShowMechanicalMappings(bool val)
{
    showMechanicalMappings_.setValue(val);
}

/// Display flags: ForceFields
void Context::setShowForceFields(bool val)
{
    showForceFields_.setValue(val);
}

/// Display flags: InteractionForceFields
void Context::setShowInteractionForceFields(bool val)
{
    showInteractionForceFields_.setValue(val);
}

/// Display flags: WireFrame
void Context::setShowWireFrame(bool val)
{
    showWireFrame_.setValue(val);
}

/// Display flags: Normals
void Context::setShowNormals(bool val)
{
    showNormals_.setValue(val);
}


// Multiresolution

bool Context::setCurrentLevel(int l)
{
    /// \TODO Shouldn't we test against finestLevel instead of 0?
    if( l > coarsestLevel_.getValue() || l < 0 ) return false;
    if( l == coarsestLevel_.getValue() )
    {
        currentLevel_.setValue(l);
        return false;
    }
    currentLevel_.setValue(l);
    return true;
}
void Context::setCoarsestLevel(int l)
{
    coarsestLevel_.setValue( l );
}
void Context::setFinestLevel(int l)
{
    finestLevel_.setValue( l );
}

//======================


void Context::copyContext(const Context& c)
{
    // BUGFIX 12/01/06 (Jeremie A.): Can't use operator= on the class as it will copy other data in the BaseContext class (such as name)...
    // *this = c;

    worldGravity_.setValue(c.worldGravity_.getValue());  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
    dt_.setValue(c.dt_.getValue());
    time_.setValue(c.time_.getValue());
    animate_.setValue(c.animate_.getValue());
    showCollisionModels_.setValue(c.showCollisionModels_.getValue());
    showBoundingCollisionModels_.setValue(c.showBoundingCollisionModels_.getValue());
    showBehaviorModels_.setValue(c.showBehaviorModels_.getValue());
    showVisualModels_.setValue(c.showVisualModels_.getValue());
    showMappings_.setValue(c.showMappings_.getValue());
    showMechanicalMappings_.setValue(c.showMechanicalMappings_.getValue());
    showForceFields_.setValue(c.showForceFields_.getValue());
    showInteractionForceFields_.setValue(c.showInteractionForceFields_.getValue());
    showWireFrame_.setValue(c.showWireFrame_.getValue());
    showNormals_.setValue(c.showNormals_.getValue());
    multiThreadSimulation_.setValue(c.multiThreadSimulation_.getValue());

    localFrame_.setValue(c.localFrame_.getValue());
    spatialVelocityInWorld_.setValue(c.spatialVelocityInWorld_.getValue());
    velocityBasedLinearAccelerationInWorld_.setValue(c.velocityBasedLinearAccelerationInWorld_.getValue());
}

std::ostream& operator << (std::ostream& out, const Context& c )
{
    std::out<<std::endl<<"local gravity = "<<c.getLocalGravity();
    std::out<<std::endl<<"transform from local to world = "<<c.getPositionInWorld();
    //std::out<<std::endl<<"transform from world to local = "<<c.getWorldToLocal();
    std::out<<std::endl<<"spatial velocity = "<<c.getVelocityInWorld();
    std::out<<std::endl<<"acceleration of the origin = "<<c.getVelocityBasedLinearAccelerationInWorld();
    return out;
}




} // namespace objectmodel

} // namespace core

} // namespace sofa

