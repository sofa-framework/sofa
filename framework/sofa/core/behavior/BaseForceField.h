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
#ifndef SOFA_CORE_BEHAVIOR_BASEFORCEFIELD_H
#define SOFA_CORE_BEHAVIOR_BASEFORCEFIELD_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace core
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
class SOFA_CORE_API BaseForceField : public virtual objectmodel::BaseObject
{
public:
    SOFA_CLASS(BaseForceField, objectmodel::BaseObject);

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
    /// $ df += kFactor K dx + bFactor B dx $
    virtual void addDForce(double kFactor = 1.0, double bFactor = 0.0) = 0;

    /// Same as addDForce(), except the velocity vector should be used instead of dx.
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ df += kFactor K v + bFactor B v $
    virtual void addDForceV(double kFactor = 1.0, double bFactor = 0.0) = 0;

    /// Accumulate the contribution of M, B, and/or K matrices multiplied
    /// by the dx vector with the given coefficients.
    ///
    /// This method computes
    /// $ df += mFactor M dx + bFactor B dx + kFactor K dx $
    /// In most cases only one of these matrices will be non-null for a given
    /// component. For forcefields without mass it simply calls addDForce.
    ///
    /// \param mFact coefficient for mass contributions (i.e. second-order derivatives term in the ODE)
    /// \param bFact coefficient for damping contributions (i.e. first derivatives term in the ODE)
    /// \param kFact coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    virtual void addMBKdx(double mFactor, double bFactor, double kFactor);

    /// Accumulate the contribution of M, B, and/or K matrices multiplied
    /// by the v vector with the given coefficients.
    ///
    /// This method computes
    /// $ df += mFactor M v + bFactor B v + kFactor K v $
    /// In most cases only one of these matrices will be non-null for a given
    /// component. For forcefields without mass it simply calls addDForceV.
    ///
    /// \param mFact coefficient for mass contributions (i.e. second-order derivatives term in the ODE)
    /// \param bFact coefficient for damping contributions (i.e. first derivatives term in the ODE)
    /// \param kFact coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    virtual void addMBKv(double mFactor, double bFactor, double kFactor);

    /// Get the potential energy associated to this ForceField.
    ///
    /// Used to extimate the total energy of the system by some
    /// post-stabilization techniques.
    virtual double getPotentialEnergy() const=0;

    /// @}

    /// @name Matrix operations
    /// @{

    /// Compute the system matrix corresponding to k K
    ///
    /// \param matrix matrix to add the result to
    /// \param kFact coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    virtual void addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double kFact) = 0;
    //virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * matrix, double kFact, unsigned int &offset);

    /// Compute the system matrix corresponding to b B
    ///
    /// \param matrix matrix to add the result to
    /// \param bFact coefficient for damping contributions (i.e. first derivatives term in the ODE)
    virtual void addBToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double bFact);
    //virtual void addBToMatrix(sofa::defaulttype::BaseMatrix * matrix, double bFact, unsigned int &offset);

    /// Compute the system matrix corresponding to m M + b B + k K
    ///
    /// \param matrix matrix to add the result to
    /// \param mFact coefficient for mass contributions (i.e. second-order derivatives term in the ODE)
    /// \param bFact coefficient for damping contributions (i.e. first derivatives term in the ODE)
    /// \param kFact coefficient for stiffness contributions (i.e. DOFs term in the ODE)
    virtual void addMBKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double mFact, double bFact, double kFact);
    //virtual void addMBKToMatrix(sofa::defaulttype::BaseMatrix * matrix, double mFact, double bFact, double kFact, unsigned int &offset);

    /// @}

    // TEMPORARY there... allow to get from the ForceField the fractured Edge index
    // When its computation is in the forcefield itself
    virtual int getFracturedEdge() {return -1;}

    /// If the forcefield is applied only on a subset of particles.
    /// That way, we can optimize the time spent to transfer forces through the mechanical mappings
    /// Deactivated by default. The forcefields using only a subset of particles should activate the mask,
    /// and during addForce(), insert the indices of the particles modified
    virtual bool useMask() {return false;}
};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
