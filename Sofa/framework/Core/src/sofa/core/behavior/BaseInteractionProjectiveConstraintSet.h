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

#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/StateAccessor.h>

namespace sofa::core::behavior
{

/**
 *  \brief BaseInteractionProjectiveConstraintSet is a constraint linking several bodies (MechanicalState) together.
 *
 *  A BaseInteractionProjectiveConstraintSet computes constraints applied to several simulated
 *  bodies given their current positions and velocities.
 *
 */
class SOFA_CORE_API BaseInteractionProjectiveConstraintSet : public BaseProjectiveConstraintSet, public virtual StateAccessor
{
public:
    SOFA_ABSTRACT_CLASS(BaseInteractionProjectiveConstraintSet, BaseProjectiveConstraintSet);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseInteractionProjectiveConstraintSet)

    /// Get the first MechanicalState
    /// \todo Rename to getMechState1()
    virtual BaseMechanicalState* getMechModel1() { return l_mechanicalStates[0]; }

    /// Get the first MechanicalState
    /// \todo Rename to getMechState2()
    virtual BaseMechanicalState* getMechModel2() { return l_mechanicalStates[1]; }


    virtual type::vector< core::BaseState* > getModels() override
    {
        return {getMechModel1(), getMechModel2() };
    }

};

} // namespace sofa::core::behavior
