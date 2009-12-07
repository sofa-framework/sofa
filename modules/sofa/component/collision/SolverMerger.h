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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_COMPONENT_COLLISION_SOLVERMERGER_H
#define SOFA_COMPONENT_COLLISION_SOLVERMERGER_H


#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <sofa/core/componentmodel/behavior/ConstraintSolver.h>
#include <sofa/helper/FnDispatcher.h>


namespace sofa
{

namespace component
{

namespace collision
{
struct SolverSet
{
    SolverSet(core::componentmodel::behavior::OdeSolver* o=NULL,core::componentmodel::behavior::LinearSolver* l=NULL,core::componentmodel::behavior::ConstraintSolver* c=NULL):
        odeSolver(o),linearSolver(l),constraintSolver(c)
    {}

    core::componentmodel::behavior::OdeSolver* odeSolver;
    core::componentmodel::behavior::LinearSolver* linearSolver;
    core::componentmodel::behavior::ConstraintSolver* constraintSolver;
};

class SolverMerger
{
public:
    static SolverSet merge(core::componentmodel::behavior::OdeSolver* solver1, core::componentmodel::behavior::OdeSolver* solver2);

protected:

    helper::FnDispatcher<core::componentmodel::behavior::OdeSolver, SolverSet> solverDispatcher;

    SolverMerger ();
};

}
}
}

#endif
