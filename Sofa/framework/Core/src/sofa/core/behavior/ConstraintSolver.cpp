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
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::core::behavior
{

ConstraintSolver::ConstraintSolver() = default;
ConstraintSolver::~ConstraintSolver() = default;

void ConstraintSolver::solveConstraint(const ConstraintParams * cParams, MultiVecId res1, MultiVecId res2)
{
    const std::string className = "SolveConstraint: " + cParams->getName();
    sofa::helper::ScopedAdvancedTimer solveConstraintTimer(className);

    bool continueSolving = true;
    {
        sofa::helper::AdvancedTimer::stepBegin(className + " - PrepareState");
        continueSolving = prepareStates(cParams, res1, res2);
        sofa::helper::AdvancedTimer::stepEnd(className + " - PrepareState");
    }

    if (continueSolving)
    {
        sofa::helper::AdvancedTimer::stepBegin(className + " - BuildSystem");
        continueSolving = buildSystem(cParams, res1, res2);
        sofa::helper::AdvancedTimer::stepEnd(className + " - BuildSystem");

        postBuildSystem(cParams);
    }

    if (continueSolving)
    {
        sofa::helper::AdvancedTimer::stepBegin(className + " - SolveSystem");
        continueSolving = solveSystem(cParams, res1, res2);
        sofa::helper::AdvancedTimer::stepEnd(className + " - SolveSystem");
    }

    if (continueSolving)
    {
        sofa::helper::AdvancedTimer::stepBegin(className + " - ApplyCorrection");
        applyCorrection(cParams, res1, res2);
        sofa::helper::AdvancedTimer::stepEnd(className + " - ApplyCorrection");
    }
}

bool ConstraintSolver::insertInNode( objectmodel::BaseNode* node )
{
    node->addConstraintSolver(this);
    Inherit1::insertInNode(node);
    return true;
}

bool ConstraintSolver::removeInNode( objectmodel::BaseNode* node )
{
    node->removeConstraintSolver(this);
    Inherit1::removeInNode(node);
    return true;
}

} // namespace sofa::core::behavior

