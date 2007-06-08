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
#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASECONSTRAINT_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_BASECONSTRAINT_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalState.h>

#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/defaulttype/SofaBaseVector.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

/**
 *  \brief Component computing constraints within a simulated body.
 *
 *  This class define the abstract API common to all constraints.
 *  A BaseConstraint computes constraints applied to one or more simulated body
 *  given its current position and velocity.
 *
 *  Constraints can be internal to a given body (attached to one MechanicalState,
 *  see the Constraint class), or link several bodies together (such as contacts,
 *  see the InteractionConstraint class).
 *
 */
class BaseConstraint : public virtual objectmodel::BaseObject
{
public:
    virtual ~BaseConstraint() { }

    /// @name Vector operations
    /// @{

    /// Project dx to constrained space (dx models an acceleration).
    virtual void projectResponse() = 0;

    /// Project dx to constrained space (dx models a velocity).
    virtual void projectVelocity() = 0;

    /// Project x to constrained space (x models a position).
    virtual void projectPosition() = 0;

    /// @}

    /// @name Matrix operations
    /// @{

    /// Project the compliance Matrix to constrained space.
    virtual void projectResponse(double **);

    /// @}

    virtual void applyConstraint(unsigned int&, double&);

    virtual void applyConstraint(defaulttype::SofaBaseMatrix *, unsigned int &);
    virtual void applyConstraint(defaulttype::SofaBaseVector *, unsigned int &);

    virtual void getConstraintValue(double * /*, unsigned int &*/) {}
    // virtual void resetContactCpt(){};

    /// Get additionnal DOFs associated to this constraint (such as Lagrange Multiplier values)
    /// \todo Remove it or disable it until we have a working Lagrange Multipliers implementation
    virtual BaseMechanicalState* getDOFs() { return NULL; }

};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
