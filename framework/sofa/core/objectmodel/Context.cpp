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
#include <sofa/core/objectmodel/Context.h>
// #include <sofa/simulation/common/Visitor.h>


namespace sofa
{

namespace core
{

namespace objectmodel
{

Context::Context()
    : is_activated(initData(&is_activated, true, "activated", "To Activate a node"))
    , worldGravity_(initData(&worldGravity_, Vec3((SReal)0,(SReal)-9.81,(SReal)0),"gravity","Gravity in the world coordinate system"))
    , dt_(initData(&dt_,0.01,"dt","Time step"))
    , time_(initData(&time_,0.,"time","Current time"))
    , animate_(initData(&animate_,false,"animate","Animate the Simulation(applied at initialization only)"))
#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MULTIRESOLUTION
    , currentLevel_(initData(&currentLevel_,0,"currentLevel","Current level of details"))
    , coarsestLevel_(initData(&coarsestLevel_,3,"coarsestLevel","Coarsest level of details"))
    , finestLevel_(initData(&finestLevel_,0,"finestLevel","Finest level of details"))
#endif
#endif // SOFA_DEV
#ifdef SOFA_SMP
    ,  processor(initData(&processor,(int )-1,"processor","assigned processor"))
    ,  gpuPrioritary(initData(&gpuPrioritary,false,"gpuPrioritary","node should be executed on GPU")),
//  is_partition_(initData(&is_partition_,false,"partition","is a parallel partition"))
    partition_(0)
#endif
{
#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MOVING_FRAMES
    setPositionInWorld(objectmodel::BaseContext::getPositionInWorld());
    setGravity(objectmodel::BaseContext::getLocalGravity());
    setVelocityInWorld(objectmodel::BaseContext::getVelocityInWorld());
    setVelocityBasedLinearAccelerationInWorld(objectmodel::BaseContext::getVelocityBasedLinearAccelerationInWorld());
#endif
#endif // SOFA_DEV

#ifdef SOFA_SMP
    is_partition_.setValue(false);
#endif
}

/// The Context is active
bool Context::isActive() const {return is_activated.getValue();}

/// State of the context
void Context::setActive(bool val)
{
    is_activated.setValue(val);
}

#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MOVING_FRAMES
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
#endif
#endif // SOFA_DEV


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


/// Gravity vector in world coordinates
const Context::Vec3& Context::getGravity() const
{
    return worldGravity_.getValue();
}



/// Animation flag
bool Context::getAnimate() const
{
    return animate_.getValue();
}

/// Display flags: Collision Models
bool Context::getShowCollisionModels() const
{
    return displayFlags_.getShowCollisionModels();
}

/// Display flags: Bounding Collision Models
bool Context::getShowBoundingCollisionModels() const
{
    return displayFlags_.getShowBoundingCollisionModels();
}

/// Display flags: Behavior Models
bool Context::getShowBehaviorModels() const
{
    return displayFlags_.getShowBehaviorModels();
}

/// Display flags: Visual Models
bool Context::getShowVisualModels() const
{
    return displayFlags_.getShowVisualModels();
}

/// Display flags: Mappings
bool Context::getShowMappings() const
{
    return displayFlags_.getShowMappings();
}

/// Display flags: Mechanical Mappings
bool Context::getShowMechanicalMappings() const
{
    return displayFlags_.getShowMechanicalMappings();
}

/// Display flags: ForceFields
bool Context::getShowForceFields() const
{
    return displayFlags_.getShowForceFields();
}

/// Display flags: InteractionForceFields
bool Context::getShowInteractionForceFields() const
{
    return displayFlags_.getShowInteractionForceFields();
}

/// Display flags: WireFrame
bool Context::getShowWireFrame() const
{
    return displayFlags_.getShowWireFrame();
}

/// Display flags: Normal
bool Context::getShowNormals() const
{
    return displayFlags_.getShowNormals();
}
#ifdef SOFA_SMP
/// Display flags: Normal
bool Context::getShowProcessorColor() const
{
    return displayFlags_.getShowProcessorColor();
}
#endif

#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MULTIRESOLUTION
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
#endif
#endif // SOFA_DEV

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
void Context::setGravity(const Vec3& g)
{
    worldGravity_ .setValue(g);
}

/// Animation flag
void Context::setAnimate(bool val)
{
    animate_.setValue(val);
}

/// Display flags: Collision Models
void Context::setShowCollisionModels(bool val)
{
    displayFlags_.setShowCollisionModels(val);
}

/// Display flags: Bounding Collision Models
void Context::setShowBoundingCollisionModels(bool val)
{
    displayFlags_.setShowBoundingCollisionModels(val);
}

/// Display flags: Behavior Models
void Context::setShowBehaviorModels(bool val)
{
    displayFlags_.setShowBehaviorModels(val);
}

/// Display flags: Visual Models
void Context::setShowVisualModels(bool val)
{
    displayFlags_.setShowVisualModels(val);

}

/// Display flags: Mappings
void Context::setShowMappings(bool val)
{
    displayFlags_.setShowMappings(val);
}

/// Display flags: Mechanical Mappings
void Context::setShowMechanicalMappings(bool val)
{
    displayFlags_.setShowMechanicalMappings(val);
}

/// Display flags: ForceFields
void Context::setShowForceFields(bool val)
{
    displayFlags_.setShowForceFields(val);
}

/// Display flags: InteractionForceFields
void Context::setShowInteractionForceFields(bool val)
{
    displayFlags_.setShowInteractionForceFields(val);
}

/// Display flags: WireFrame
void Context::setShowWireFrame(bool val)
{
    displayFlags_.setShowWireFrame(val);
}

/// Display flags: Normals
void Context::setShowNormals(bool val)
{
    displayFlags_.setShowNormals(val);
}

#ifdef SOFA_SMP
void Context::setShowProcessorColor(bool val)
{
    displayFlags_.setShowProcessorColor(val);
}
#endif

#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MULTIRESOLUTION
// Multiresolution

bool Context::setCurrentLevel(int l)
{
    if( l > coarsestLevel_.getValue() )
    {
        currentLevel_.setValue(coarsestLevel_.getValue());
        return false;
    }
    else if( l < 0 /*finestLevel_.getValue()*/ )
    {
// 		currentLevel_.setValue(finestLevel_.getValue());
        currentLevel_.setValue( 0 );
        return false;
    }
    currentLevel_.setValue(l);
    if( l == coarsestLevel_.getValue() ) return false;
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
#endif
#endif // SOFA_DEV

//======================


void Context::copyContext(const Context& c)
{
    // BUGFIX 12/01/06 (Jeremie A.): Can't use operator= on the class as it will copy other data in the BaseContext class (such as name)...
    // *this = c;

    copySimulationContext(c);
    copyVisualContext(c);
}
#ifdef SOFA_SMP
int Context::getProcessor() const
{
    return processor.getValue();
}
void Context::setProcessor(int p)
{
    processor.setValue(p);
}
#endif


void Context::copySimulationContext(const Context& c)
{
    worldGravity_.setValue(c.getGravity());  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
    setDt(c.getDt());
    setTime(c.getTime());
    setAnimate(c.getAnimate());
#ifdef SOFA_SMP
    if(c.gpuPrioritary.getValue())
        gpuPrioritary.setValue(true);
#endif

#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MOVING_FRAMES
    setPositionInWorld( c.getPositionInWorld());
    spatialVelocityInWorld_ = c.spatialVelocityInWorld_;
    velocityBasedLinearAccelerationInWorld_ = c.velocityBasedLinearAccelerationInWorld_;
#endif
#endif // SOFA_DEV

#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MULTIRESOLUTION
    // for multiresolution
// 	finestLevel_ = c.finestLevel_;
// 	coarsestLevel_ = c.coarsestLevel_;
// 	currentLevel_ = c.currentLevel_;
#endif
#endif // SOFA_DEV

#ifdef SOFA_SMP
    if(!partition_)
    {
        if(processor.getValue()!=-1)
            is_partition_.setValue(true);
        if(is_partition())
        {

            partition_= new Iterative::IterativePartition();
//          partition_->setCPU(processor.getValue());
        }
    }
    if(processor.getValue()==-1&&c.processor.getValue()!=-1)
    {
        processor.setValue(c.processor.getValue());
        is_partition_.setValue(true);
    }
    if(c.is_partition()&&!partition_)
    {
        partition_=c.getPartition();
        is_partition_.setValue(true);
    }
    if((gpuPrioritary.getValue())&&partition_)
    {
        partition_->setGPUPrioritary();
    }

#endif

}

void Context::copyVisualContext(const Context& c)
{
    setShowCollisionModels(c.getShowCollisionModels());
    setShowBoundingCollisionModels(c.getShowBoundingCollisionModels());
    setShowBehaviorModels(c.getShowBehaviorModels());
    setShowVisualModels(c.getShowVisualModels());
    setShowMappings(c.getShowMappings());
    setShowMechanicalMappings(c.getShowMechanicalMappings());
    setShowForceFields(c.getShowForceFields());
    setShowInteractionForceFields(c.getShowInteractionForceFields());
    setShowWireFrame(c.getShowWireFrame());
    setShowNormals(c.getShowNormals());
#ifdef SOFA_SMP
    setShowProcessorColor(c.getShowProcessorColor());
#endif
}


void Context::fusionVisualContext(const Context& c)
{
    setShowCollisionModels(getShowCollisionModels() || c.getShowCollisionModels());
    setShowBoundingCollisionModels(getShowBoundingCollisionModels() || c.getShowBoundingCollisionModels());
    setShowBehaviorModels(getShowBehaviorModels() || c.getShowBehaviorModels());
    setShowVisualModels(getShowVisualModels() || c.getShowVisualModels());
    setShowMappings(getShowMappings() || c.getShowMappings());
    setShowMechanicalMappings(getShowMechanicalMappings() || c.getShowMechanicalMappings());
    setShowForceFields(getShowForceFields() || c.getShowForceFields());
    setShowInteractionForceFields(getShowInteractionForceFields() || c.getShowInteractionForceFields());
    setShowWireFrame(getShowWireFrame() || c.getShowWireFrame());
    setShowNormals(getShowNormals() || c.getShowNormals());
}




} // namespace objectmodel

} // namespace core

} // namespace sofa

