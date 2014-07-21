/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_BASEINTERACTIONPROJECTIVECONSTRAINTSET_H
#define SOFA_CORE_BEHAVIOR_BASEINTERACTIONPROJECTIVECONSTRAINTSET_H

#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief BaseInteractionProjectiveConstraintSet is a constraint linking several bodies (MechanicalState) together.
 *
 *  A BaseInteractionProjectiveConstraintSet computes constraints applied to several simulated
 *  bodies given their current positions and velocities.
 *
 */
class SOFA_CORE_API BaseInteractionProjectiveConstraintSet : public BaseProjectiveConstraintSet
{
public:
    SOFA_ABSTRACT_CLASS(BaseInteractionProjectiveConstraintSet, BaseProjectiveConstraintSet);

    /// Get the first MechanicalState
    /// \todo Rename to getMechState1()
    /// \todo Replace with an accessor to a list of states, as an InteractionConstraint can be applied to more than two.
    virtual BaseMechanicalState* getMechModel1() = 0;

    /// Get the first MechanicalState
    /// \todo Rename to getMechState2()
    /// \todo Replace with an accessor to a list of states, as an InteractionConstraint can be applied to more than two.
    virtual BaseMechanicalState* getMechModel2() = 0;


    virtual helper::vector< core::BaseState* > getModels()
    {
        helper::vector< core::BaseState* > models;
        models.push_back( getMechModel1() );
        models.push_back( getMechModel2() );
        return models;
    }

};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_BEHAVIOR_BASEINTERACTIONPROJECTIVECONSTRAINTSET_H
