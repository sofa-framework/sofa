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
#include <sofa/component/odesolver/backward/config.h>

#include <sofa/core/behavior/LinearSolverAccessor.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/BaseIntegrationMethod.h>

namespace sofa::component::odesolver::backward
{

class SOFA_COMPONENT_ODESOLVER_BACKWARD_API NewtonRaphsonSolver
    : public sofa::core::behavior::OdeSolver
    , public sofa::core::behavior::LinearSolverAccessor
{
public:
    SOFA_CLASS2(NewtonRaphsonSolver, sofa::core::behavior::OdeSolver, sofa::core::behavior::LinearSolverAccessor);

    void init() override;
    void solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    SingleLink<NewtonRaphsonSolver, core::behavior::BaseIntegrationMethod, BaseLink::FLAG_STRONGLINK> l_integrationMethod;

    Data<SReal> d_absoluteResidualToleranceThreshold;

protected:
    NewtonRaphsonSolver();
};

}
