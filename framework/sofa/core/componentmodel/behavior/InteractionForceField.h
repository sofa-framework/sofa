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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_INTERACTIONFORCEFIELD_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_INTERACTIONFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/BaseForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/defaulttype/SofaBaseVector.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

/**
 *  \brief InteractionForceField is a force field linking several bodies (MechanicalState) together.
 *
 *  An interaction force field computes forces applied to several simulated
 *  bodies given their current positions and velocities.
 *
 *  For implicit integration schemes, it must also compute the derivative
 *  ( df, given a displacement dx ).
 */
class InteractionForceField : public BaseForceField
{
public:

    /// Get the first MechanicalState
    /// \todo Rename to getMechState1()
    /// \todo Replace with an accessor to a list of states, as an InteractionForceField can be applied to more than two.
    virtual BaseMechanicalState* getMechModel1() = 0;

    /// Get the first MechanicalState
    /// \todo Rename to getMechState2()
    /// \todo Replace with an accessor to a list of states, as an InteractionForceField can be applied to more than two.
    virtual BaseMechanicalState* getMechModel2() = 0;


    /// @name Matrix operations
    /// @{

    virtual void contributeToMatrixDimension(unsigned int * const, unsigned int * const) {};
    virtual void computeMatrix(sofa::defaulttype::SofaBaseMatrix *, double , double , double, unsigned int &) {};
    virtual void computeVector(sofa::defaulttype::SofaBaseVector *, unsigned int &) {};
    virtual void matResUpdatePosition(sofa::defaulttype::SofaBaseVector *, unsigned int &) {};

    /// @}
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
