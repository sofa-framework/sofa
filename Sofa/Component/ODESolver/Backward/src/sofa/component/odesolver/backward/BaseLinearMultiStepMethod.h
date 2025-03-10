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
#include <sofa/core/behavior/LinearSolverAccessor.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

namespace sofa::component::odesolver::backward
{

class SOFA_COMPONENT_ODESOLVER_BACKWARD_API BaseLinearMultiStepMethod :
    public core::behavior::OdeSolver,
    public core::behavior::LinearSolverAccessor
{
public:
    SOFA_CLASS2(BaseLinearMultiStepMethod, core::behavior::OdeSolver,
                core::behavior::LinearSolverAccessor);

    BaseLinearMultiStepMethod();

    void init() override;
    void reset() override;
    void solve(const core::ExecParams* params, SReal dt,
        core::MultiVecCoordId xResult,
        core::MultiVecDerivId vResult) final;

    Data<std::size_t> d_order;

    Data<SReal> d_rayleighStiffness;
    Data<SReal> d_rayleighMass;

    SingleLink<BaseLinearMultiStepMethod, NewtonRaphsonSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_newtonSolver;

protected:

    template <core::VecType vtype>
    void realloc(sofa::simulation::common::VectorOperations& vop,
        core::TMultiVecId<vtype, core::VecAccess::V_WRITE>& id,
        const std::string& vecIdName)
    {
        sofa::core::behavior::TMultiVec<vtype> vec(&vop, id);
        vec.realloc(&vop, false, true, core::VecIdProperties{vecIdName, this->getClassName()});
        id = vec.id();
    }

    std::deque<SReal> m_timeList;

    std::size_t m_currentSolve { 0 };

    sofa::type::vector<SReal> m_a_coef;
    sofa::type::vector<SReal> m_b_coef;

    virtual void recomputeCoefficients(std::size_t order, SReal dt) = 0;

private:

    sofa::core::MultiVecDerivId m_r1, m_r2, m_rhs, m_dv;

    sofa::type::vector<core::MultiVecCoordId> m_position;
    sofa::type::vector<core::MultiVecDerivId> m_velocity;
    sofa::type::vector<core::MultiVecDerivId> m_force;

};



}  // namespace sofa::component::odesolver::backward
