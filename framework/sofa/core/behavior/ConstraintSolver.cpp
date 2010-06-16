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
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{

namespace core
{

namespace behavior
{

ConstraintSolver::ConstraintSolver()
{}

ConstraintSolver::~ConstraintSolver()
{}

void ConstraintSolver::solveConstraint(double dt, VecId id, ConstOrder order)
{
    sofa::helper::AdvancedTimer::stepBegin("SolveConstraints");
    bool continueSolving=true;
    sofa::helper::AdvancedTimer::stepBegin("SolveConstraints PrepareState");
    continueSolving=prepareStates(dt, id, order);
    sofa::helper::AdvancedTimer::stepEnd  ("SolveConstraints PrepareState");
    if (continueSolving)
    {
        sofa::helper::AdvancedTimer::stepBegin("SolveConstraints BuildSystem");
        continueSolving=buildSystem(dt, id, order);
        sofa::helper::AdvancedTimer::stepEnd  ("SolveConstraints BuildSystem");
    }
    else
    {
        sofa::helper::AdvancedTimer::stepEnd  ("SolveConstraints");
        return;
    }
    if (continueSolving)
    {
        sofa::helper::AdvancedTimer::stepBegin("SolveConstraints SolveSystem");
        continueSolving=solveSystem(dt, id, order);
        sofa::helper::AdvancedTimer::stepEnd  ("SolveConstraints SolveSystem");
    }
    else
    {
        sofa::helper::AdvancedTimer::stepEnd  ("SolveConstraints");
        return;
    }

    if (continueSolving)
    {
        sofa::helper::AdvancedTimer::stepBegin("SolveConstraints ApplyCorrection");
        continueSolving=applyCorrection(dt, id, order);
        sofa::helper::AdvancedTimer::stepEnd  ("SolveConstraints ApplyCorrection");
    }

    sofa::helper::AdvancedTimer::stepEnd("SolveConstraints");
}
} // namespace behavior

} // namespace core

} // namespace sofa

