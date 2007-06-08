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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEFORCEFIELD_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASEFORCEFIELD_H

#include <sofa/core/objectmodel/BaseObject.h>
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
 *  \brief Component computing forces within a simulated body.
 *
 *  This class define the abstract API common to all force fields.
 *  A force field computes forces applied to one or more simulated body
 *  given its current position and velocity.
 *
 *  Forces can be internal to a given body (attached to one MechanicalState,
 *  see the ForceField class), or link several bodies together (such as contact
 *  forces, see the InteractionForceField class).
 *
 *  For implicit integration schemes, it must also compute the derivative
 *  ( df, given a displacement dx ).
 *
 */
class BaseForceField : public virtual objectmodel::BaseObject
{
public:
    virtual ~BaseForceField() {}

    /// @name Vector operations
    /// @{

    /// Given the current position and velocity states, update the current force
    /// vector by computing and adding the forces associated with this
    /// ForceField.
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ f += B v + K x $
    virtual void addForce() = 0;

    /// Compute the force derivative given a small displacement from the
    /// position and velocity used in the previous call to addForce().
    ///
    /// The derivative should be directly derived from the computations
    /// done by addForce. Any forces neglected in addDForce will be integrated
    /// explicitly (i.e. using its value at the beginning of the timestep).
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ df += K dx $
    virtual void addDForce() = 0;

    /// Get the potential energy associated to this ForceField.
    ///
    /// Used to extimate the total energy of the system by some
    /// post-stabilization techniques.
    virtual double getPotentialEnergy() =0;

    /// @}

    /// @name Matrix operations
    /// @{

    /// Compute the system matrix corresponding to m M + b B + k K
    ///
    /// \param m coefficient for mass values
    /// \param b coefficient for damping values
    /// \param k coefficient for stiffness values
    /// \param offset current row/column offset, must be incremented
    ///   by this method
    virtual void computeMatrix(sofa::defaulttype::SofaBaseMatrix * matrix, double mFact, double bFact, double kFact, unsigned int &offset);

    /// Compute the system matrix dimmensions bo adding the number of lines and
    /// columns associated with this ForceField.
    ///
    /// \todo Isn't the dimensions related to the MechanicalState and not the
    /// ForceFields ?
    virtual void contributeToMatrixDimension(unsigned int * const nbRow, unsigned int * const nbCol);

    /// Compute the right-hand side vector of the system matrix.
    virtual void computeVector(sofa::defaulttype::SofaBaseVector * vect, unsigned int &offset);

    virtual void matResUpdatePosition(sofa::defaulttype::SofaBaseVector * vect, unsigned int &offset);

    /// @}
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
