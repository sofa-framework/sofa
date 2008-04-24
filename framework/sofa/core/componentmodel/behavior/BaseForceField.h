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
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

/**
 *  \brief Component computing forces within simulated bodies.
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
    typedef sofa::defaulttype::Vector3::value_type Real_Sofa;
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

    /// Same as addDForce(), except the velocity vector should be used instead of dx.
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ df += K v $
    virtual void addDForceV() = 0;

    /// Get the potential energy associated to this ForceField.
    ///
    /// Used to extimate the total energy of the system by some
    /// post-stabilization techniques.
    virtual Real_Sofa getPotentialEnergy() =0;

    /// @}

    /// @name Matrix operations
    /// @{

    /// Compute the system matrix corresponding to m M
    ///
    /// \param matrix matrix to add the result to
    /// \param mFact coefficient for mass values
    /// \param offset current row/column offset
    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * matrix, Real_Sofa kFact, unsigned int &offset);

    /// Compute the right-hand side vector of the system matrix.
    virtual void addKDxToVector(sofa::defaulttype::BaseVector * vect, Real_Sofa kFact, unsigned int &offset);

    /// @}

    // TEMPORARY there... allow to get from the ForceField the fractured Edge index
    // When its computation is in the forcefield itself
    virtual int getFracturedEdge() {return -1;};
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
