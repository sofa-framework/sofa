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
#ifndef SOFA_CORE_OBJECTMODEL_BASECONTEXT_H
#define SOFA_CORE_OBJECTMODEL_BASECONTEXT_H

#include <sofa/core/objectmodel/Base.h>
#include <sofa/defaulttype/SolidTypes.h>
//#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <set>

namespace sofa
{
namespace simulation
{
namespace tree
{
class Visitor;
}
}

namespace core
{

namespace objectmodel
{

class BaseObject;
class Event;

/**
 *  \brief Base class for Context classes, storing shared variables and parameters.
 *
 *  A Context contains values or pointers to variables and parameters shared
 *  by a group of objects, typically refering to the same simulated body.
 *  Derived classes can defined simple isolated contexts or more powerful
 *  hierarchical representations (scene-graphs), in which case the context also
 *  implements the BaseNode interface.
 *
 * \author Jeremie Allard
 */
class BaseContext : public virtual Base
{
public:

    /// @name Types defined for local coordinate system handling
    /// @{
    typedef defaulttype::SolidTypes<double> SolidTypes;
    typedef SolidTypes::Transform Frame;
    typedef SolidTypes::Vec Vec3;
    typedef SolidTypes::Rot Quat;
    typedef SolidTypes::Mat Mat33;
    typedef SolidTypes::SpatialVector SpatialVector;
    /// @}

    BaseContext();
    virtual ~BaseContext();

    /// Get the default Context object, that contains the default values for
    /// all parameters and can be used when no local context is defined.
    static BaseContext* getDefault();

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

    /// Multiresolution support (UNSTABLE)
    virtual int getCurrentLevel() const;

    /// Multiresolution support (UNSTABLE)
    virtual int getCoarsestLevel() const;

    /// Multiresolution support (UNSTABLE)
    virtual int getFinestLevel() const;

    /// Multiresolution support (UNSTABLE)
    virtual unsigned int nbLevels() const;

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
    ///// Gravity in local coordinates
    //virtual void setGravity( const Vec3& ) { }
    /// Gravity in world coordinates
    virtual const Vec3& getGravityInWorld() const;
    /// Gravity in world coordinates
    virtual void setGravityInWorld( const Vec3& )
    { }

    /// @name Variables
    /// @{

    /// Mechanical Degrees-of-Freedom
    virtual BaseObject* getMechanicalState() const;

    /// Topology
    virtual BaseObject* getTopology() const;

    /// Dynamic Topology
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

    /// Multiresolution support (UNSTABLE) : Set the current level, return false if l >= coarsestLevel
    virtual bool setCurrentLevel(int )
    {
        return false;
    }

    /// Multiresolution support (UNSTABLE)
    virtual void setCoarsestLevel(int ) {}

    /// Multiresolution support (UNSTABLE)
    virtual void setFinestLevel(int ) {}

    /// @}

    /// @name Variables Setters
    /// @{

    /// Mechanical Degrees-of-Freedom
    virtual void setMechanicalState( BaseObject* )
    { }

    /// Topology
    virtual void setTopology( BaseObject* )
    { }

    /// @}

    /// @name Adding/Removing objects. Note that these methods can fail if the context doesn't support attached objects
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

    /// @name Visitors.
    /// @{

    /// apply an action
    virtual void executeVisitor( simulation::tree::Visitor* );

    /// Propagate an event
    virtual void propagateEvent( Event* );

    /// @}

};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif


