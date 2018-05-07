/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

namespace sofa
{

namespace core
{

namespace behavior
{

ConstraintSolver::ConstraintSolver()
    : m_fId(VecDerivId::externalForce())
    , m_dxId(VecDerivId::dx())
{}

ConstraintSolver::~ConstraintSolver()
{}

void ConstraintSolver::solveConstraint(const ConstraintParams * cParams, MultiVecId res1, MultiVecId res2)
{
    using sofa::helper::AdvancedTimer;

    std::string className = "SolveConstraints " + cParams->getName();
    AdvancedTimer::stepBegin(className + "SolveConstraints ");

    AdvancedTimer::stepBegin(className + " PrepareState");
    bool continueSolving = prepareStates(cParams, res1, res2);
    AdvancedTimer::stepEnd(className + " PrepareState");

    if (continueSolving)
    {
        AdvancedTimer::stepBegin(className + " BuildSystem");
        continueSolving = buildSystem(cParams, res1, res2);
        AdvancedTimer::stepEnd(className + " BuildSystem");
    }
    else
    {
        AdvancedTimer::stepEnd(className);
        return;
    }

    if (continueSolving)
    {
        AdvancedTimer::stepBegin(className + " SolveSystem ");
        continueSolving = solveSystem(cParams, res1, res2);
        AdvancedTimer::stepEnd(className + " SolveSystem ");
    }
    else
    {
        AdvancedTimer::stepEnd(className);
        return;
    }

    if (continueSolving)
    {
        AdvancedTimer::stepBegin(className + " ApplyCorrection ");
        applyCorrection(cParams, res1, res2);
        AdvancedTimer::stepEnd(className + " ApplyCorrection ");
    }

    AdvancedTimer::stepEnd(className + "SolveConstraints ");
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




} // namespace behavior

} // namespace core

} // namespace sofa

