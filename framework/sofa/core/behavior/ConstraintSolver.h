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
#ifndef SOFA_CORE_BEHAVIOR_CONSTRAINTSOLVER_H
#define SOFA_CORE_BEHAVIOR_CONSTRAINTSOLVER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief Component responsible for the expression and solution of system of equations related to constraints
 *
 */
class SOFA_CORE_API ConstraintSolver : public virtual objectmodel::BaseObject
{
    typedef ConstraintParams::ConstOrder ConstOrder;

public:
    SOFA_CLASS(ConstraintSolver, objectmodel::BaseObject);

    ConstraintSolver();
    virtual ~ConstraintSolver();

    /** Launch the sequence of operations in order to solve the constraints
     * @param Id order of the constraint to be solved
     * @param isPositionChangesUpdateVelocity boolean indication if we need to propagate the change of position to a modification of velocity dv=dx/dt
     **/
    virtual void solveConstraint(double /*dt*/, VecId, ConstOrder order);



    /**
     * Do the precomputation: compute free state, or propagate the states to the mapped mechanical states, where the constraint can be expressed
     */
    virtual bool prepareStates(double /*dt*/, VecId, ConstOrder order)=0;

    /**
     * Create the system corresponding to the constraints
     */
    virtual bool buildSystem(double /*dt*/, VecId, ConstOrder order)=0;

    /**
     * Use the system previously built and solve it with the appropriate algorithm
     */
    virtual bool solveSystem(double /*dt*/, VecId, ConstOrder order)=0;

    /**
     * Correct the Mechanical State with the solution found
     */
    virtual bool applyCorrection(double /*dt*/, VecId, ConstOrder order)=0;
};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
