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
#ifndef SOFA_CORE_OBJECTMODEL_CONTEXT_H
#define SOFA_CORE_OBJECTMODEL_CONTEXT_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Data.h>

#include <iostream>
#include <map>

#ifdef SOFA_SMP
#include <IterativePartition.h>
#include <AthapascanIterative.h>
#endif


namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 *  \brief Implementation of BaseContext, storing all shared parameters in Datas.
 *
 */
class SOFA_CORE_API Context : public BaseContext
{
public:
    SOFA_CLASS(Context, BaseContext);



    Data<bool> is_activated;
    Data<Vec3> worldGravity_;  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
    Data<double> dt_;
    Data<double> time_;
    Data<bool> animate_;
#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MULTIRESOLUTION
    /// @name For multiresolution (UNSTABLE)
    /// @{
    Data<int> currentLevel_;
    Data<int> coarsestLevel_;
    Data<int> finestLevel_;
    /// @}
#endif
#endif // SOFA_DEV

#ifdef SOFA_SMP
    Data<int> processor;
    Data<bool> gpuPrioritary;
    Data<bool> is_partition_;
    Iterative::IterativePartition *partition_;
#endif

protected:
    Context();
    virtual ~Context()
    {}
public:

    /// @name Parameters
    /// @{

    /// The Context is active
    virtual bool isActive() const;
    /// State of the context
    virtual void setActive(bool val);

    /// Gravity in local coordinates
    virtual const Vec3& getGravity() const;
    /// Gravity in local coordinates
    virtual void setGravity( const Vec3& );

    /// Simulation timestep
    virtual double getDt() const;

    /// Simulation time
    virtual double getTime() const;

    /// Animation flag
    virtual bool getAnimate() const;

#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MULTIRESOLUTION
    /// Multiresolution support (UNSTABLE)
    virtual int getCurrentLevel() const;
    /// Multiresolution support (UNSTABLE)
    virtual int getCoarsestLevel() const;
    /// Multiresolution support (UNSTABLE)
    virtual int getFinestLevel() const;
#endif
#endif // SOFA_DEV

    /// @}


#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MOVING_FRAMES
    /// @name Local Coordinate System
    /// @{
    typedef BaseContext::Frame Frame;
    typedef BaseContext::Vec3 Vec3;
    typedef BaseContext::Quat Quat;
    typedef BaseContext::SpatialVector SpatialVector;

    Frame localFrame_;
    SpatialVector spatialVelocityInWorld_;
    Vec3 velocityBasedLinearAccelerationInWorld_;
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

    /// Gravity in the local coordinate system  TODO: replace with world coordinates
    virtual Vec3 getLocalGravity() const;
    /// Gravity in the local coordinate system
    //virtual void setGravity(const Vec3& );
    /// @}
#endif
#endif // SOFA_DEV

    /// @name Parameters Setters
    /// @{

    /// Simulation timestep
    virtual void setDt( double dt );

    /// Simulation time
    virtual void setTime( double t );

    /// Animation flag
    virtual void setAnimate(bool val);

    /// Display flags: Gravity
    virtual void setDisplayWorldGravity(bool val) { worldGravity_.setDisplayed(val); }

#ifdef SOFA_DEV
#ifdef SOFA_SUPPORT_MULTIRESOLUTION
    /// Multiresolution support (UNSTABLE) : Set the current level, return false if l >= coarsestLevel
    virtual bool setCurrentLevel(int l);
    /// Multiresolution support (UNSTABLE)
    virtual void setCoarsestLevel(int l);
    /// Multiresolution support (UNSTABLE)
    virtual void setFinestLevel(int l);
#endif
#endif // SOFA_DEV

#ifdef SOFA_SMP
    inline bool is_partition()const {return is_partition_.getValue();}
    inline Iterative::IterativePartition *getPartition()const {return partition_;}
    /// Accessor to the object processor
    int getProcessor() const;
    void setProcessor(int);
#endif

    /// @}

    /// Copy the context variables from the given instance
    void copyContext(const Context& c);

    /// Copy the context variables of visualization from the given instance
    void copySimulationContext(const Context& c);



};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
