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

#include "sofa/simulation/MechanicalOperations.h"
#include "sofa/simulation/VectorOperations.h"

namespace sofa::component::odesolver::backward
{

template<class T>
concept isLinearMultiStepMethodParameters = requires
{
    std::is_convertible_v<decltype(T::Order), std::size_t>;
    std::is_convertible_v<decltype(T::a_coef), sofa::type::fixed_array<SReal, T::Order>>;
    std::is_convertible_v<decltype(T::b_coef), sofa::type::fixed_array<SReal, T::Order>>;
};

class LinearMultiStepMethodResidual : public newton_raphson::BaseNonLinearFunction
{
public:
    ~LinearMultiStepMethodResidual() override = default;
    void evaluateCurrentGuess() override {}
    SReal squaredNormLastEvaluation() override { return 0; }
    void computeGradientFromCurrentGuess() override {}
    void updateGuessFromLinearSolution() override {}
    void solveLinearEquation() override {}
};

template <isLinearMultiStepMethodParameters LinearMultiStepMethodParameters>
class BaseLinearMultiStepMethod : public sofa::core::behavior::OdeSolver
{
   public:
    void solve(const core::ExecParams* params, SReal dt,
        core::MultiVecCoordId xResult,
        core::MultiVecDerivId vResult) final
    {
        if (!isComponentStateValid())
        {
            return;
        }

        // Create the vector and mechanical operations tools. These are used to execute special
        // operations (multiplication,
        // additions, etc.) on multi-vectors (a vector that is stored in different buffers inside
        // the mechanical objects)
        sofa::simulation::common::VectorOperations vop(params, this->getContext());
        sofa::simulation::common::MechanicalOperations mop(params, this->getContext());
        mop->setImplicit(true);

        // LinearMultiStepMethodResidual residual;
        // LinearMultiStepMethodVector initialGuess;
        // l_newtonSolver->solve(residual, initialGuess);
    }

    void init() override;

    SingleLink<BaseLinearMultiStepMethod, NewtonRaphsonSolver,
               BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK>
        l_newtonSolver;
};

template <isLinearMultiStepMethodParameters LinearMultiStepMethodParameters>
void BaseLinearMultiStepMethod<LinearMultiStepMethodParameters>::init()
{
    OdeSolver::init();

    if (!l_newtonSolver.get())
    {
        l_newtonSolver.set(getContext()->get<NewtonRaphsonSolver>(getContext()->getTags(), core::objectmodel::BaseContext::SearchDown));

        if (!l_newtonSolver)
        {
            msg_error() << "A Newton-Raphson solver (" << NewtonRaphsonSolver::GetClass()->className
                << ") is required by this component but has not been found.";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }
    }
}

}  // namespace sofa::component::odesolver::backward
