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

/**
 * Base class for a linear multistep method
 *
 * Generic class computing a residual based on any coefficients given by a linear multistep method.
 * Derived class just have to define the coefficients.
 *
 * A linear multistep method is defined by:
 * $$
 * \sum_{j=0}^s a_j y_{n+j} = h\sum_{j=0}^s b_j f(t_{n+j},y_{n+j})
 * $$
 *
 * The coefficients a and b are class members and must be computed by derived classes.
 *
 * The linear system from the Newton-Raphson solver is solved in 2 steps:
 * 1) A first linear system leads to solve for a velocity difference
 * 2) Based on the velocity, the position is computed
 */
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

    // list of coefficients of the linear multistep method corresponding to a in the formula
    sofa::type::vector<SReal> m_a_coef;
    // list of coefficients of the linear multistep method corresponding to b in the formula
    sofa::type::vector<SReal> m_b_coef;

    // recompute coefficients a and b
    virtual void recomputeCoefficients(std::size_t order, SReal dt) = 0;

private:

    /**
     * Multiple vectors must be defined:
     * - The residual is divided into two terms r1 and r2
     * - The right-hand side of the linear system
     * - The solution of the linear system (dv)
     */
    sofa::core::MultiVecDerivId m_r1, m_r2, m_rhs, m_dv;

    // depending on the order of the method, position must be stored from previous time steps
    sofa::type::vector<core::MultiVecCoordId> m_position;
    // depending on the order of the method, velocity must be stored from previous time steps
    sofa::type::vector<core::MultiVecDerivId> m_velocity;
    // depending on the order of the method, force must be stored from previous time steps
    sofa::type::vector<core::MultiVecDerivId> m_force;

};



}  // namespace sofa::component::odesolver::backward
