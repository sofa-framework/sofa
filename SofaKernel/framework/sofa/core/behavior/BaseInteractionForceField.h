/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_BEHAVIOR_BASEINTERACTIONFORCEFIELD_H
#define SOFA_CORE_BEHAVIOR_BASEINTERACTIONFORCEFIELD_H

#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/MechanicalParams.h>
namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief BaseInteractionForceField is a force field linking several bodies (MechanicalState) together.
 *
 *  An interaction force field computes forces applied to several simulated
 *  bodies given their current positions and velocities.
 *
 *  For implicit integration schemes, it must also compute the derivative
 *  ( df, given a displacement dx ).
 */
class SOFA_CORE_API BaseInteractionForceField : public BaseForceField
{
public:
    SOFA_ABSTRACT_CLASS(BaseInteractionForceField, BaseForceField);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseInteractionForceField)

    /// Get the first MechanicalState
    /// \todo Rename to getMechState1()
    /// \todo Replace with an accessor to a list of states, as an InteractionForceField can be applied to more than two.
    virtual BaseMechanicalState* getMechModel1() = 0;

    /// Get the first MechanicalState
    /// \todo Rename to getMechState2()
    /// \todo Replace with an accessor to a list of states, as an InteractionForceField can be applied to more than two.
    virtual BaseMechanicalState* getMechModel2() = 0;

    virtual void addKToMatrix(const MechanicalParams* /* mparams */, const sofa::core::behavior::MultiMatrixAccessor* /* matrix */ ) override
    {
        serr << "ERROR("<<getClassName()<<"): addKToMatrix not implemented." << sendl;
    }


    /// initialization to export potential energy to gnuplot files format
    virtual void initGnuplot(const std::string path){
        sout << path << std::endl << "Warning ::: initGnuplot not implemented for all interaction force field" << sendl;
    }

    /// export kinetic and potential energy state at "time" to a gnuplot file
    virtual void exportGnuplot(SReal time){
        sout << time << std::endl << "Warning ::: exportGnuplot not implemented for all interaction force field" << sendl;
    }


    virtual bool insertInNode( objectmodel::BaseNode* node ) override;
    virtual bool removeInNode( objectmodel::BaseNode* node ) override;

};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_BEHAVIOR_BASEINTERACTIONFORCEFIELD_H
