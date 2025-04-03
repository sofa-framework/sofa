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

#include <sofa/component/odesolver/backward/NewtonRaphsonSolver.h>
#include <sofa/component/odesolver/backward/config.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/LinearSolverAccessor.h>

namespace sofa::component::odesolver::backward
{

class SOFA_COMPONENT_ODESOLVER_BACKWARD_API StaticOdeSolver :
    public core::behavior::OdeSolver,
    public core::behavior::LinearSolverAccessor
{
public:
    SOFA_CLASS2(StaticOdeSolver, core::behavior::OdeSolver, core::behavior::LinearSolverAccessor);
    StaticOdeSolver();

    void solve(
        const core::ExecParams* params,
        SReal dt,
        core::MultiVecCoordId xResult,
        core::MultiVecDerivId vResult) override;

    void init() override;

    SingleLink<StaticOdeSolver, NewtonRaphsonSolver,
               BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK>
        l_newtonSolver;

    auto squared_residual_norms() const -> const std::vector<SReal> & = delete;
    auto squared_increment_norms() const -> const std::vector<SReal> & = delete;

protected:

    struct NewtonRaphsonDeprecatedData : core::objectmodel::lifecycle::DeprecatedData
    {
        NewtonRaphsonDeprecatedData(Base* b, const std::string name)
            : DeprecatedData(b, "v25.06", "v25.12", name, "The Data related to the Newton-Raphson parameters must be defined in the NewtonRaphsonSolver component.")
        {}
    };

    SOFA_ATTRIBUTE_DEPRECATED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData d_newton_iterations;
    SOFA_ATTRIBUTE_DEPRECATED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData d_absolute_correction_tolerance_threshold;
    SOFA_ATTRIBUTE_DEPRECATED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData d_relative_correction_tolerance_threshold;
    SOFA_ATTRIBUTE_DEPRECATED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData d_absolute_residual_tolerance_threshold;
    SOFA_ATTRIBUTE_DEPRECATED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData d_relative_residual_tolerance_threshold;
    SOFA_ATTRIBUTE_DEPRECATED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData d_should_diverge_when_residual_is_growing;
};

}
