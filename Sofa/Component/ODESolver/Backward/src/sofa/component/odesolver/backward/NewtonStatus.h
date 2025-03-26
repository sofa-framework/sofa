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
#include <sofa/helper/SelectableItem.h>
#include <array>

namespace sofa::component::odesolver::backward
{
MAKE_SELECTABLE_ITEMS(NewtonStatus,
    sofa::helper::Item{"Undefined", "The solver has not been called yet"},
    sofa::helper::Item{"Running", "The solver is still running and/or did not finish"},
    sofa::helper::Item{"ConvergedEquilibrium", "Converged: the iterations did not start because the system is already at equilibrium"},
    sofa::helper::Item{"DivergedLineSearch", "Diverged: line search failed"},
    sofa::helper::Item{"DivergedMaxIterations", "Diverged: Reached the maximum number of iterations"},
    sofa::helper::Item{"ConvergedResidualSuccessiveRatio", "Converged: Residual successive ratio is smaller than the threshold"},
    sofa::helper::Item{"ConvergedResidualInitialRatio", "Converged: Residual initial ratio is smaller than the threshold"},
    sofa::helper::Item{"ConvergedAbsoluteResidual", "Converged: Absolute residual is smaller than the threshold"},
    sofa::helper::Item{"ConvergedRelativeEstimateDifference", "Converged: Relative estimate difference is smaller than the threshold"},
    sofa::helper::Item{"ConvergedAbsoluteEstimateDifference", "Converged: Absolute estimate difference is smaller than the threshold"});
}
