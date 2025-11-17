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

#include <sofa/component/constraint/lagrangian/solver/BuiltConstraintSolver.h>
#include <sofa/core/behavior/ConstraintResolution.h>

namespace sofa::component::constraint::lagrangian::solver
{
class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API ProjectedGaussSeidelConstraintSolver : public BuiltConstraintSolver
{
public:
    SOFA_CLASS(ProjectedGaussSeidelConstraintSolver, BuiltConstraintSolver);

protected:
    virtual void doSolve(GenericConstraintProblem * problem , SReal timeout = 0.0) override;

    void gaussSeidel_increment(bool measureError, SReal *dfree, SReal *force, SReal **w, SReal tol, SReal *d, int dim, bool& constraintsAreVerified, SReal& error, std::vector<core::behavior::ConstraintResolution*>& constraintCorrections, sofa::type::vector<SReal>& tabErrors) const;

};
}