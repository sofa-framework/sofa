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
#include <SofaMiscCollision/config.h>

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/helper/FnDispatcher.h>

namespace sofa::component::collision
{

/// A structure containing pointers to an ODE solver, a linear solver and a constraint solver
/// Any could be nullptr
struct SOFA_MISC_COLLISION_API SolverSet
{
    explicit SolverSet(core::behavior::OdeSolver::SPtr o = nullptr,
                       core::behavior::BaseLinearSolver::SPtr l = nullptr,
                       core::behavior::ConstraintSolver::SPtr c = nullptr);

    core::behavior::OdeSolver::SPtr odeSolver;
    core::behavior::BaseLinearSolver::SPtr linearSolver;
    core::behavior::ConstraintSolver::SPtr constraintSolver;
};

/// Helper class aiming at creating a common ODE solver, linear solver and constraint solver, based on
/// a pair of ODE solvers
/// The rules of creating the common solvers depends on the type of the provided ODE solvers (see SolverMerger()).
/// For example: EulerImplicitSolver + EulerExplicitSolver -> EulerImplicitSolver
class SOFA_MISC_COLLISION_API SolverMerger
{
public:

    /// Static function to call to create a common solver set based on a pair of ODE solvers
    static SolverSet merge(core::behavior::OdeSolver* solver1, core::behavior::OdeSolver* solver2);

    template<typename SolverType1, typename SolverType2, SolverSet (*F)(SolverType1&,SolverType2&),bool symmetric> static void addDispatcher()
    {
        getInstance()->solverDispatcher.add<SolverType1,SolverType2,F,symmetric>();
    }

protected:

    static SolverMerger* getInstance();

    helper::FnDispatcher<core::behavior::OdeSolver, SolverSet> solverDispatcher;

    SolverMerger ();
};

} //namespace sofa::component::collision