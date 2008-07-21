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

#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{
/// @TODO  The classes applyConstraint, getConstraintValue && getConstraintId need to be commented
#ifdef SOFA_DEV
/**
 *  \brief Object computing a constraint resolution within a Gauss-Seidel algorithm
 */

class ConstraintResolution
{
public:
    ConstraintResolution()
        : nbLines(1) {}

    virtual ~ConstraintResolution() {}

    virtual void init(int /*line*/, double** /*w*/) {}
    virtual void resolution(int line, double** w, double* d, double* force) = 0;

    unsigned char nbLines;
};
#endif // SOFA_DEV

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

    /// Project v to constrained space (v models a velocity).
    virtual void projectVelocity() = 0;

    /// Project x to constrained space (x models a position).
    virtual void projectPosition() = 0;

    /// Project vFree to constrained space (vFree models a velocity).
    virtual void projectFreeVelocity() = 0;

    /// Project xFree to constrained space (xFree models a position).
    virtual void projectFreePosition() = 0;

    /// @}

    /// @name Matrix operations
    /// @{

    /// Project the compliance Matrix to constrained space.
    virtual void projectResponse(double **);

    /// @}

    virtual void applyConstraint(unsigned int&, double&);

    /// Project the global Mechanical Matrix to constrained space using offset parameter
    virtual void applyConstraint(defaulttype::BaseMatrix *, unsigned int & /*offset*/);

    /// Project the global Mechanical Vector to constrained space using offset parameter
    virtual void applyConstraint(defaulttype::BaseVector *, unsigned int & /*offset*/);

    virtual void getConstraintValue(defaulttype::BaseVector *) {};
    virtual void getConstraintValue(double *) {};
    virtual void getConstraintId(long * /*id*/, unsigned int & /*offset*/) {}

#ifdef SOFA_DEV
    /// Add the corresponding ConstraintResolution using the offset parameter
    virtual void getConstraintResolution(std::vector<ConstraintResolution*>& /*resTab*/, unsigned int& /*offset*/) {};
#endif //SOFA_DEV

    /// Get additionnal DOFs associated to this constraint (such as Lagrange Multiplier values)
    /// \todo Remove it or disable it until we have a working Lagrange Multipliers implementation
    virtual BaseMechanicalState* getDOFs() { return NULL; }

    /// says if the constraint is holonomic or not
    /// holonomic constraints can be processed using different methods such as :
    /// projection - reducing the degrees of freedom - simple lagrange multiplier process
    /// Non-holonomic constraints (like contact, friction...) need more specific treatments
    virtual bool isHolonomic() {return false; }

};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
