/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/BaseComponent.h>

namespace sofa::core::objectmodel
{

/**
 *  \brief Compatibility layer - Context is now a subclass of BaseContext.
 *  All functionality has been merged into BaseContext.
 *  This class is kept for backward compatibility with existing code.
 */
class SOFA_CORE_API Context : public BaseContext
{
public:
    SOFA_CLASS(Context, BaseContext);

    // All Data members are inherited from BaseContext
    using BaseContext::is_activated;
    using BaseContext::worldGravity_;
    using BaseContext::dt_;
    using BaseContext::time_;
    using BaseContext::animate_;
    using BaseContext::d_isSleeping;
    using BaseContext::d_canChangeSleepingState;

protected:
    Context();
    virtual ~Context() override = default;

public:
    /// @name Parameters
    /// @{

    /// The Context is active
    bool isActive() const override;
    /// State of the context
    void setActive(bool val) override;

    /// The Context is sleeping
    bool isSleeping() const override;

    /// The Context can change its sleeping state
    bool canChangeSleepingState() const override;

    /// Gravity in local coordinates
    const Vec3& getGravity() const override;
    /// Gravity in local coordinates
    void setGravity( const Vec3& ) override;

    /// Simulation timestep
    SReal getDt() const override;

    /// Simulation time
    SReal getTime() const override;

    /// Animation flag
    bool getAnimate() const override;
    /// @}

    /// @name Parameters Setters
    /// @{

    /// Simulation timestep
    void setDt( SReal dt ) override;

    /// Simulation time
    virtual void setTime( SReal t );

    /// Animation flag
    void setAnimate(bool val) override;

    /// Sleeping state of the context
    void setSleeping(bool val) override;

    /// Sleeping state change of the context
    void setChangeSleepingState(bool val) override;

    /// Display flags: Gravity
    virtual void setDisplayWorldGravity(bool val);

    /// @}

    /// Copy the context variables from the given instance
    void copyContext(const Context& c);

    /// Copy the context variables of visualization from the given instance
    void copySimulationContext(const Context& c);

};
} // namespace sofa::core::objectmodel
