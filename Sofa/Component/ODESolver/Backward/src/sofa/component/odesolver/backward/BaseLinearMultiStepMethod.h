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

template<class T>
concept isLinearMultiStepMethodParameters = requires
{
    std::is_convertible_v<decltype(T::Order), std::size_t>;
    std::is_convertible_v<decltype(T::a_coef), sofa::type::fixed_array<SReal, T::Order + 1>>;
    std::is_convertible_v<decltype(T::b_coef), sofa::type::fixed_array<SReal, T::Order + 1>>;
};

template <isLinearMultiStepMethodParameters LinearMultiStepMethodParameters>
class BaseLinearMultiStepMethod : public core::behavior::OdeSolver,
                                  public core::behavior::LinearSolverAccessor
{
public:
    SOFA_CLASS2(BaseLinearMultiStepMethod, core::behavior::OdeSolver,
                core::behavior::LinearSolverAccessor);

    static constexpr auto Order = LinearMultiStepMethodParameters::Order;
    static constexpr bool IsImplicit = LinearMultiStepMethodParameters::b_coef[Order] != 0;

    BaseLinearMultiStepMethod();

    void init() override;
    void solve(const core::ExecParams* params, SReal dt,
        core::MultiVecCoordId xResult,
        core::MultiVecDerivId vResult) final;

    Data<SReal> d_rayleighStiffness;
    Data<SReal> d_rayleighMass;

    SingleLink<BaseLinearMultiStepMethod, NewtonRaphsonSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_newtonSolver;

protected:
    template <core::VecType vtype>
    void realloc(sofa::simulation::common::VectorOperations& vop,
        core::TMultiVecId<vtype, core::VecAccess::V_WRITE>& id,
        const std::string& vecIdName);

private:

    sofa::core::MultiVecDerivId m_r1, m_r2, m_rhs, m_dv;

    sofa::type::fixed_array<core::MultiVecCoordId, Order + 1> m_position;
    sofa::type::fixed_array<core::MultiVecDerivId, Order + 1> m_velocity;
    sofa::type::fixed_array<core::MultiVecDerivId, Order + 1> m_force;

    static constexpr auto& a_coef = LinearMultiStepMethodParameters::a_coef;
    static constexpr auto& b_coef = LinearMultiStepMethodParameters::b_coef;

    struct ResidualFunction : newton_raphson::BaseNonLinearFunction
    {
        void evaluateCurrentGuess() override
        {
            {
                SCOPED_TIMER("ComputeForce");
                static constexpr bool clearForcesBeforeComputingThem = true;
                static constexpr bool applyBottomUpMappings = true;

                mop.mparams.setX(position[Order]);
                mop.mparams.setV(velocity[Order]);
                mop.computeForce(force[Order], clearForcesBeforeComputingThem, applyBottomUpMappings);

                //Rayleigh damping
                mop.addMBKv(force[Order],
                    core::MatricesFactors::M(-rayleighMass),
                    core::MatricesFactors::B(0),
                    core::MatricesFactors::K(rayleighStiffness));

                mop.projectResponse(force[Order]);
            }

            core::behavior::MultiVecDeriv r1Vec(&vop, r1);
            core::behavior::MultiVecDeriv r2Vec(&vop, r2);

            r1Vec.clear();
            r2Vec.clear();

            for (unsigned int i = 0; i < Order + 1; ++i)
            {
                if (a_coef[i] != 0)
                {
                    r1Vec.peq(position[i], a_coef[i]);
                }
            }
            for (unsigned int i = 0; i < Order + 1; ++i)
            {
                if (b_coef[i] != 0)
                {
                    r1Vec.peq(velocity[i], -dt * b_coef[i]);
                }
            }

            core::behavior::MultiVecDeriv tmp(&vop);
            for (unsigned int i = 0; i < Order + 1; ++i)
            {
                if (a_coef[i] != 0)
                {
                    tmp.peq(velocity[i], a_coef[i]);
                }
            }
            mop.addMdx(r2, tmp, core::MatricesFactors::M(1).get());

            for (unsigned int i = 0; i < Order + 1; ++i)
            {
                if (b_coef[i] != 0)
                {
                    r2Vec.peq(force[i], -dt * b_coef[i]);
                }
            }

            mop.projectResponse(r1);
            mop.projectResponse(r2);
        }

        SReal squaredNormLastEvaluation() override
        {
            core::behavior::MultiVecDeriv r2Vec(&vop, r2);
            return r2Vec.dot(r2Vec);
        }

        void computeGradientFromCurrentGuess() override
        {
            const core::MatricesFactors::M m(a_coef[Order] + dt * b_coef[Order] * rayleighMass);
            const core::MatricesFactors::B b(-dt * b_coef[Order]);
            const core::MatricesFactors::K k(-dt * b_coef[Order] * (rayleighStiffness + dt * b_coef[Order] / a_coef[Order]));

            mop.setSystemMBKMatrix(m, b, k, linearSolver);

            core::behavior::MultiVecDeriv rhsVec(&vop, rhs);
            rhsVec.eq(r2, -1);

            // rhs += -dt * (bs /as) * K * r1
            {
                const auto vBackup = mop.mparams.v();
                mop.mparams.setV(r1);
                mop.addMBKv(rhs, core::MatricesFactors::M(0), core::MatricesFactors::B(0),
                    core::MatricesFactors::K(-dt * b_coef[Order] / a_coef[Order]));
                mop.mparams.setV(vBackup);
            }

            mop.projectResponse(rhs);
        }

        void updateGuessFromLinearSolution() override
        {
            vop.v_peq(velocity[Order], dv);
            vop.v_peq(position[Order], dx);

            mop.projectPositionAndVelocity(position[Order], velocity[Order]);
            mop.propagateXAndV(position[Order], velocity[Order]);
        }

        void solveLinearEquation() override
        {
            vop.v_clear(dv);

            linearSolver->setSystemLHVector(dv);
            linearSolver->setSystemRHVector(rhs);
            linearSolver->solveSystem();

            core::behavior::MultiVecDeriv dxVec(&vop, dx);
            dxVec.eq(dv, dt * b_coef[Order] / a_coef[Order]);
            dxVec.peq(r1, - 1 / a_coef[Order]);
        }

        SReal dt = 0;
        SReal rayleighStiffness = 0;
        SReal rayleighMass = 0;
        sofa::simulation::common::MechanicalOperations& mop;
        simulation::common::VectorOperations& vop;
        sofa::type::fixed_array<core::MultiVecCoordId, Order + 1> position;
        sofa::type::fixed_array<core::MultiVecDerivId, Order + 1> velocity;
        sofa::type::fixed_array<core::MultiVecDerivId, Order + 1> force;
        core::MultiVecDerivId r1;
        core::MultiVecDerivId r2;
        core::MultiVecDerivId dx;
        core::MultiVecDerivId dv;
        core::MultiVecDerivId rhs;
        core::behavior::LinearSolver* linearSolver { nullptr };

        ResidualFunction(
            sofa::simulation::common::MechanicalOperations& mop,
            simulation::common::VectorOperations& vop)
        : mop(mop), vop(vop)
        {}

        ResidualFunction& setRayleighStiffness(SReal v)
        {
            rayleighStiffness = v;
            return *this;
        }

        ResidualFunction& setRayleighMass(SReal v)
        {
            rayleighMass = v;
            return *this;
        }

        ResidualFunction& setDt(SReal v)
        {
            dt = v;
            return *this;
        }

        ResidualFunction& setPosition(const sofa::type::fixed_array<core::MultiVecCoordId, Order + 1>& v)
        {
            position = v;
            return *this;
        }

        ResidualFunction& setVelocity(const sofa::type::fixed_array<core::MultiVecDerivId, Order + 1>& v)
        {
            velocity = v;
            return *this;
        }

        ResidualFunction& setForce(const sofa::type::fixed_array<core::MultiVecDerivId, Order + 1>& v)
        {
            force = v;
            return *this;
        }

        ResidualFunction& setR1(core::MultiVecDerivId v)
        {
            r1 = v;
            return *this;
        }

        ResidualFunction& setR2(core::MultiVecDerivId v)
        {
            r2 = v;
            return *this;
        }

        ResidualFunction& setDx(core::MultiVecDerivId v)
        {
            dx = v;
            return *this;
        }

        ResidualFunction& setDv(core::MultiVecDerivId v)
        {
            dv = v;
            return *this;
        }

        ResidualFunction& setRHS(core::MultiVecDerivId v)
        {
            rhs = v;
            return *this;
        }

        ResidualFunction& setLinearSolver(core::behavior::LinearSolver* solver)
        {
            linearSolver = solver;
            return *this;
        }

    };
};


template <isLinearMultiStepMethodParameters LinearMultiStepMethodParameters>
BaseLinearMultiStepMethod<LinearMultiStepMethodParameters>::BaseLinearMultiStepMethod()
    : d_rayleighStiffness(initData(&d_rayleighStiffness, 0_sreal, "rayleighStiffness",
                                   "Rayleigh damping coefficient related to stiffness, > 0")),
      d_rayleighMass(initData(&d_rayleighMass, 0_sreal, "rayleighMass",
                              "Rayleigh damping coefficient related to mass, > 0")),
      l_newtonSolver(initLink("newtonSolver", "Link to a Newton-Raphson solver to solve the nonlinear equation produced by this numerical method"))
{
}

template <isLinearMultiStepMethodParameters LinearMultiStepMethodParameters>
void BaseLinearMultiStepMethod<LinearMultiStepMethodParameters>::init()
{
    OdeSolver::init();
    LinearSolverAccessor::init();

    if (!l_newtonSolver.get())
    {
        l_newtonSolver.set(getContext()->get<NewtonRaphsonSolver>(
            getContext()->getTags(), core::objectmodel::BaseContext::SearchDown));

        if (!l_newtonSolver)
        {
            msg_error()
                << "A Newton-Raphson solver is required by this component but has not been found.";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }
    }

    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Invalid)
    {
        d_componentState.setValue(core::objectmodel::ComponentState::Valid);
    }

    simulation::common::VectorOperations vop(sofa::core::execparams::defaultInstance(),
                                             this->getContext());

    for (unsigned int i = 0; i < Order + 1; ++i)
    {
        if (i != Order - 1)
        {
            realloc(vop, m_position[i], "x+" + std::to_string(i));
            realloc(vop, m_velocity[i], "v+" + std::to_string(i));
            realloc(vop, m_force[i], "f+" + std::to_string(i));
        }
    }

    realloc(vop, m_r1, "r1");
    realloc(vop, m_r2, "r2");
    realloc(vop, m_rhs, "rhs");
    realloc(vop, m_dv, "dv");
}

template <isLinearMultiStepMethodParameters LinearMultiStepMethodParameters>
template <core::VecType vtype>
void BaseLinearMultiStepMethod<LinearMultiStepMethodParameters>::realloc(
    sofa::simulation::common::VectorOperations& vop,
    core::TMultiVecId<vtype, core::VecAccess::V_WRITE>& id,
    const std::string& vecIdName)
{
    sofa::core::behavior::TMultiVec<vtype> vec(&vop, id);
    vec.realloc(&vop, false, true, core::VecIdProperties{vecIdName, this->getClassName()});
    id = vec.id();
}

template <isLinearMultiStepMethodParameters LinearMultiStepMethodParameters>
void BaseLinearMultiStepMethod<LinearMultiStepMethodParameters>::solve(
    const core::ExecParams* params, SReal dt,
    core::MultiVecCoordId xResult,
    core::MultiVecDerivId vResult)
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
    mop->setImplicit(IsImplicit);

    {
        core::behavior::MultiVecDeriv dx(&vop, sofa::core::vec_id::write_access::dx);
        dx.realloc(&vop, false, true);
    }

    m_position[Order-1] = xResult;
    m_velocity[Order-1] = vResult;
    m_force[Order-1] = sofa::core::vec_id::write_access::force;

    for (unsigned int i = 0; i < Order + 1; ++i)
    {
        if (i != Order - 1)
        {
            realloc(vop, m_position[i], "x+" + std::to_string(i));
            realloc(vop, m_velocity[i], "v+" + std::to_string(i));
            realloc(vop, m_force[i], "f+" + std::to_string(i));
        }
    }

    realloc(vop, m_r1, "r1");
    realloc(vop, m_r2, "r2");
    realloc(vop, m_rhs, "rhs");
    realloc(vop, m_dv, "dv");

    //initial guess
    vop.v_eq(m_position[Order], m_position[Order - 1]);
    vop.v_eq(m_velocity[Order], m_velocity[Order - 1]);

    ResidualFunction residualFunction(mop, vop);
    residualFunction
        .setDt(dt)
        .setRayleighMass(d_rayleighMass.getValue())
        .setRayleighStiffness(d_rayleighStiffness.getValue())
        .setPosition(m_position)
        .setVelocity(m_velocity)
        .setForce(m_force)
        .setR1(m_r1)
        .setR2(m_r2)
        .setDx(sofa::core::vec_id::write_access::dx)
        .setDv(m_dv)
        .setRHS(m_rhs)
        .setLinearSolver(l_linearSolver.get());
    l_newtonSolver->solve(residualFunction);

    if (f_printLog.getValue())
    {
        for (unsigned int i = 0; i < Order + 1; ++i)
        {
            core::behavior::MultiVecDeriv v(&vop, m_velocity[i]);
            std::cout << "v[" << i << "] = " << v << std::endl;

            core::behavior::MultiVecCoord x(&vop, m_position[i]);
            std::cout << "x[" << i << "] = " << x << std::endl;
        }
    }

    //update state
    for (unsigned int i = 0; i < Order; ++i)
    {
        vop.v_eq(m_velocity[i], m_velocity[i+1]);
        vop.v_eq(m_position[i], m_position[i+1]);
        vop.v_eq(m_force[i], m_force[i+1]);
    }
}


}  // namespace sofa::component::odesolver::backward
