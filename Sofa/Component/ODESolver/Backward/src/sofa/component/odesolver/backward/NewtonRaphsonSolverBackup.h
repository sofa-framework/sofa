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
#include <sofa/component/odesolver/backward/NewtonStatus.h>
#include <sofa/component/odesolver/backward/config.h>
#include <sofa/core/behavior/LinearSolverAccessor.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>

#include "convergence/NewtonRaphsonConvergenceMeasure.h"
#include <sofa/helper/map.h>

namespace sofa::component::odesolver::backward
{

template<class Derived>
struct StateVersionAccess
{
    int id {};
    constexpr explicit StateVersionAccess(int i) : id(i) {}
    constexpr StateVersionAccess() = default;

    constexpr Derived operator+(const int v) const
    {
        return Derived(id + v);
    }

    constexpr Derived operator-(const int v) const
    {
        return Derived(id - v);
    }
};

struct TimeStepStateVersionAccess : StateVersionAccess<TimeStepStateVersionAccess>
{
    using StateVersionAccess::StateVersionAccess;
};

struct NewtonIterationStateVersionAccess : StateVersionAccess<NewtonIterationStateVersionAccess>
{
    using StateVersionAccess::StateVersionAccess;
};

template <core::VecType vtype>
struct StateList
{
    using MultiVec = core::behavior::TMultiVec<vtype>;

    // list of states needed by the numerical integration of ODE
    std::deque<MultiVec> timeStepStates;

    // used by the Newton-Raphson algorithm to store an intermediate vector
    std::deque<MultiVec> newtonIterationStates;

    MultiVec& operator[](const TimeStepStateVersionAccess& id) 
    {
        assert(id.id <= 0);
        // n is the last element of the states and correspond to the state from the previous time step
        // For example, for a 2-step method, the state list will look like: [n-1, n]
        const auto n = timeStepStates.size() - 1;
        return timeStepStates[n + id.id];
    }

    MultiVec& operator[](const NewtonIterationStateVersionAccess& id)
    {
        assert(id.id >= 0);
        // i is the first element of the states and correspond to the state from the previous iteration
        const auto i = 0;
        return newtonIterationStates[i + id.id];
    }

    void setOps(core::behavior::BaseVectorOperations* op)
    {
        for (auto& state : timeStepStates)
        {
            state.realloc(op);
        }

        for (auto& state : newtonIterationStates)
        {
            state.realloc(op);
        }
    }
};

class SOFA_COMPONENT_ODESOLVER_BACKWARD_API NewtonRaphsonSolverBackup
    : public sofa::core::behavior::OdeSolver
    , public sofa::core::behavior::LinearSolverAccessor
{
public:
    SOFA_CLASS2(NewtonRaphsonSolverBackup, sofa::core::behavior::OdeSolver, sofa::core::behavior::LinearSolverAccessor);

    ~NewtonRaphsonSolverBackup() override;

    void init() override;
    void reset() override;
    void solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    SingleLink<NewtonRaphsonSolverBackup, core::behavior::BaseIntegrationMethod, BaseLink::FLAG_STRONGLINK> l_integrationMethod;

    Data<unsigned int> d_maxNbIterationsNewton;
    Data<SReal> d_relativeSuccessiveStoppingThreshold;
    Data<SReal> d_relativeInitialStoppingThreshold;
    Data<SReal> d_absoluteResidualStoppingThreshold;
    Data<SReal> d_relativeEstimateDifferenceThreshold;
    Data<SReal> d_absoluteEstimateDifferenceThreshold;
    Data<unsigned int> d_maxNbIterationsLineSearch;
    Data<SReal> d_lineSearchCoefficient;
    Data<bool> d_updateStateWhenDiverged;
    Data<NewtonStatus> d_status;
    Data<std::map < std::string, sofa::type::vector<SReal> > > d_residualGraph;

protected:
    NewtonRaphsonSolverBackup();

    core::behavior::MultiVecDeriv m_linearSystemSolution;

    void computeRightHandSide(const core::ExecParams* params, SReal dt,
                      core::MultiVecDerivId force,
                      core::MultiVecDerivId b,
                      core::MultiVecDerivId velocity_i,
                      core::MultiVecCoordId position_i) const;


    SReal computeResidual(const core::ExecParams* params,
                          sofa::simulation::common::MechanicalOperations& mop, SReal dt,
                          core::MultiVecDerivId force, core::MultiVecDerivId oldVelocity,
                          core::MultiVecDerivId newVelocity);
    
    void resizeStateList(std::size_t nbStates, sofa::simulation::common::VectorOperations& vop);
    void start();

    StateList<core::V_COORD> m_coordStates;
    StateList<core::V_DERIV> m_derivStates;
    
    bool measureConvergence(
        const NewtonRaphsonConvergenceMeasure& measure,
        sofa::simulation::common::VectorOperations& vop,
        sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult,
        std::stringstream& os);

};

}
