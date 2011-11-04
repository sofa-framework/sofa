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

    /// Get the first MechanicalState
    /// \todo Rename to getMechState1()
    /// \todo Replace with an accessor to a list of states, as an InteractionForceField can be applied to more than two.
    virtual BaseMechanicalState* getMechModel1() = 0;

    /// Get the first MechanicalState
    /// \todo Rename to getMechState2()
    /// \todo Replace with an accessor to a list of states, as an InteractionForceField can be applied to more than two.
    virtual BaseMechanicalState* getMechModel2() = 0;

    virtual void addKToMatrix(const MechanicalParams* mparams /* PARAMS FIRST */, const sofa::core::behavior::MultiMatrixAccessor* matrix)
    {
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef r1 = matrix->getMatrix(getMechModel1());
        sofa::core::behavior::MultiMatrixAccessor::MatrixRef r2 = matrix->getMatrix(getMechModel2());
        if (r1) addKToMatrix(r1.matrix, mparams->kFactor(), r1.offset);
        if (r2) addKToMatrix(r2.matrix,  mparams->kFactor(), r2.offset);
    }

    /// @deprecated
    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * /*matrix*/, double /*kFact*/, unsigned int &/*offset*/)
    {
        serr << "ERROR("<<getClassName()<<"): addKToMatrix not implemented." << sendl;
    }

};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_BEHAVIOR_BASEINTERACTIONFORCEFIELD_H
