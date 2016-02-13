/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_SOLVERMERGER_H
#define SOFA_COMPONENT_COLLISION_SOLVERMERGER_H
#include "config.h"

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/helper/FnDispatcher.h>


namespace sofa
{

namespace component
{

namespace collision
{
struct SolverSet
{
    SolverSet(core::behavior::OdeSolver::SPtr o=NULL,core::behavior::BaseLinearSolver::SPtr l=NULL,core::behavior::ConstraintSolver::SPtr c=NULL):
        odeSolver(o),linearSolver(l),constraintSolver(c)
    {}

    core::behavior::OdeSolver::SPtr odeSolver;
    core::behavior::BaseLinearSolver::SPtr linearSolver;
    core::behavior::ConstraintSolver::SPtr constraintSolver;
};

class SOFA_MISC_COLLISION_API SolverMerger
{
public:
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

}
}
}

#endif
