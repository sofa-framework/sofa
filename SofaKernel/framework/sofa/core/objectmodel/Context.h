/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_OBJECTMODEL_CONTEXT_H
#define SOFA_CORE_OBJECTMODEL_CONTEXT_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseContext.h>



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



    Data<bool> is_activated; ///< To Activate a node
    Data<Vec3> worldGravity_;  ///< Gravity IN THE WORLD COORDINATE SYSTEM.
    Data<SReal> dt_; ///< Time step
    Data<SReal> time_; ///< Current time
    Data<bool> animate_; ///< Animate the Simulation(applied at initialization only)
	Data<bool> d_isSleeping;				///< Tells if the context is sleeping, and thus ignored by visitors
	Data<bool> d_canChangeSleepingState;	///< Tells if the context can change its sleeping state



protected:
    Context();
    virtual ~Context()
    {}
public:

    /// @name Parameters
    /// @{

    /// The Context is active
    virtual bool isActive() const override;
    /// State of the context
    virtual void setActive(bool val) override;

	/// The Context is sleeping
	virtual bool isSleeping() const override;

	/// The Context can change its sleeping state
	virtual bool canChangeSleepingState() const override;

    /// Gravity in local coordinates
    virtual const Vec3& getGravity() const override;
    /// Gravity in local coordinates
    virtual void setGravity( const Vec3& ) override;

    /// Simulation timestep
    virtual SReal getDt() const override;

    /// Simulation time
    virtual SReal getTime() const override;

    /// Animation flag
    virtual bool getAnimate() const override;
    /// @}

    /// @name Parameters Setters
    /// @{

    /// Simulation timestep
    virtual void setDt( SReal dt ) override;

    /// Simulation time
    virtual void setTime( SReal t );

    /// Animation flag
    virtual void setAnimate(bool val) override;

	/// Sleeping state of the context
	virtual void setSleeping(bool val) override;

	/// Sleeping state change of the context
	virtual void setChangeSleepingState(bool val) override;

    /// Display flags: Gravity
    virtual void setDisplayWorldGravity(bool val) { worldGravity_.setDisplayed(val); }

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
