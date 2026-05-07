/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/core/behavior/ConstraintResolution.h>



namespace sofa::component::constraint::lagrangian::solver
{

/**
 *  \brief This component implements a generic way of preparing system for solvers that doesn't need
 *  a build version of the constraint matrix. Any solver that are based on an unbuilt system should
 *  inherit from this.
 *  This component is purely virtual because doSolve is not defined and needs to be defined in the
 *  inherited class
 */
class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API UnbuiltConstraintSolver : public GenericConstraintSolver
{
public:
    SOFA_CLASS(UnbuiltConstraintSolver, GenericConstraintSolver);


    virtual void initializeConstraintProblems() override;

protected:
    virtual void doBuildSystem( const core::ConstraintParams *cParams, GenericConstraintProblem * problem, unsigned int numConstraints) override;
};
}