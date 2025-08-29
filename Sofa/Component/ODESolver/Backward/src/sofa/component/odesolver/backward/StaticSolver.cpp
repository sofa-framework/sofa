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
#include <sofa/component/odesolver/backward/StaticSolver.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>

namespace sofa::component::odesolver::backward
{

void registerStaticSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(
        core::ObjectRegistrationData("Static ODE Solver")
            .add<StaticSolver>());
}

StaticSolver::StaticSolver()
    : l_newtonSolver(initLink("newtonSolver", "Link to a NewtonRaphsonSolver"))
    , d_newton_iterations(this, "newton_iterations")
    , d_absolute_correction_tolerance_threshold(this, "absolute_correction_tolerance_threshold")
    , d_relative_correction_tolerance_threshold(this, "relative_correction_tolerance_threshold")
    , d_absolute_residual_tolerance_threshold(this, "absolute_residual_tolerance_threshold")
    , d_relative_residual_tolerance_threshold(this, "relative_residual_tolerance_threshold")
    , d_should_diverge_when_residual_is_growing(this, "should_diverge_when_residual_is_growing")
{}

void StaticSolver::parse(core::objectmodel::BaseObjectDescription* arg)
{
    Inherit1::parse(arg);

    const auto warnNewAttribute = [&, arg](auto& data, const std::string& newAttributeName)
    {
        if (const char* attribute = arg->getAttribute(data.m_name))
        {
            data.value.emplace(std::stod(attribute));
            msg_warning() << "The attribute '" << data.m_name
                << "' is no longer defined in this component. Instead, define the attribute '"
                << newAttributeName << "' in the NewtonRaphsonSolver component associated with this StaticSolver.";
        }
    };

    warnNewAttribute(d_newton_iterations, "maxNbIterationsNewton");
    warnNewAttribute(d_absolute_correction_tolerance_threshold, "absoluteEstimateDifferenceThreshold");
    warnNewAttribute(d_relative_correction_tolerance_threshold, "relativeEstimateDifferenceThreshold");
    warnNewAttribute(d_absolute_residual_tolerance_threshold, "absoluteResidualStoppingThreshold");
    warnNewAttribute(d_relative_residual_tolerance_threshold, "relativeEstimateDifferenceThreshold");
}

void StaticSolver::init()
{
    OdeSolver::init();
    LinearSolverAccessor::init();

    if (!l_newtonSolver.get())
    {
        l_newtonSolver.set(getContext()->get<NewtonRaphsonSolver>(getContext()->getTags(), core::objectmodel::BaseContext::SearchDown));

        if (!l_newtonSolver)
        {
            msg_warning() << "A Newton-Raphson solver is required by this component but has not been found. It will be created automatically";
            auto newtonRaphsonSolver = core::objectmodel::New<NewtonRaphsonSolver>();
            newtonRaphsonSolver->setName(this->getContext()->getNameHelper().resolveName(newtonRaphsonSolver->getClassName(), core::ComponentNameHelper::Convention::xml));
            this->getContext()->addObject(newtonRaphsonSolver);
            l_newtonSolver.set(newtonRaphsonSolver);

            const auto setDeprecatedAttribute = [&]<class T1, class T2>(const NewtonRaphsonDeprecatedData<T1>& oldData, Data<T2>& newData)
            {
                if (oldData.value.has_value())
                {
                    newData.setValue(*oldData.value);
                    msg_warning() << "The attribute '" << newData.getName() << "' in " << newData.getOwner()->getPathName()
                        << " is set from the deprecated attribute '" << oldData.m_name << "'. This will be removed in the future.";
                }
            };

            setDeprecatedAttribute(d_newton_iterations, l_newtonSolver->d_maxNbIterationsNewton);
            setDeprecatedAttribute(d_absolute_correction_tolerance_threshold, l_newtonSolver->d_absoluteEstimateDifferenceThreshold);
            setDeprecatedAttribute(d_relative_correction_tolerance_threshold, l_newtonSolver->d_relativeEstimateDifferenceThreshold);
            setDeprecatedAttribute(d_absolute_residual_tolerance_threshold, l_newtonSolver->d_absoluteResidualStoppingThreshold);
            setDeprecatedAttribute(d_relative_residual_tolerance_threshold, l_newtonSolver->d_relativeEstimateDifferenceThreshold);

        }
    }

    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Invalid)
    {
        d_componentState.setValue(core::objectmodel::ComponentState::Valid);
    }
}

struct StaticResidualFunction : newton_raphson::BaseNonLinearFunction
{
    void evaluateCurrentGuess() override
    {
        SCOPED_TIMER("ComputeForce");
        static constexpr bool clearForcesBeforeComputingThem = true;
        static constexpr bool applyBottomUpMappings = true;

        mop.computeForce(force, clearForcesBeforeComputingThem, applyBottomUpMappings);
        mop.projectResponse(force);
    }

    SReal squaredNormLastEvaluation() override
    {
        return force.dot(force);
    }

    void computeGradientFromCurrentGuess() override
    {
        SCOPED_TIMER("MBKBuild");

        static constexpr core::MatricesFactors::M m(0);
        static constexpr core::MatricesFactors::B b(0);
        static constexpr core::MatricesFactors::K k(-1);

        mop.setSystemMBKMatrix(m, b, k, linearSolver);
    }

    void updateGuessFromLinearSolution(SReal alpha) override
    {
        x.peq(dx, alpha);
        mop.solveConstraint(x, sofa::core::ConstraintOrder::POS);
        mop.propagateX(x);
    }

    void solveLinearEquation() override
    {
        SCOPED_TIMER("MBKSolve");

        linearSolver->getLinearSystem()->setSystemSolution(dx);
        linearSolver->getLinearSystem()->setRHS(force);
        linearSolver->solveSystem();
        linearSolver->getLinearSystem()->dispatchSystemSolution(dx);
    }

    SReal squaredNormDx() override
    {
        return dx.dot(dx);
    }

    SReal squaredLastEvaluation() override
    {
        return x.dot(x);
    }

    sofa::simulation::common::MechanicalOperations& mop;
    core::behavior::MultiVecCoord& x;
    core::behavior::MultiVecDeriv& force;
    core::behavior::MultiVecDeriv& dx;
    core::behavior::LinearSolver* linearSolver { nullptr };

    StaticResidualFunction(sofa::simulation::common::MechanicalOperations& mop,
                           core::behavior::MultiVecCoord& x, core::behavior::MultiVecDeriv& force,
                           core::behavior::MultiVecDeriv& dx,
                           core::behavior::LinearSolver* linearSolver)
        : mop(mop), x(x), force(force), dx(dx), linearSolver(linearSolver)
    {
    }
};

void StaticSolver::solve(const core::ExecParams* params, SReal dt, core::MultiVecCoordId xResult,
                         core::MultiVecDerivId vResult)
{
    if (!isComponentStateValid())
    {
        return;
    }

    SOFA_UNUSED(dt);
    SOFA_UNUSED(vResult);

    // Create the vector and mechanical operations tools. These are used to execute special
    // operations (multiplication, additions, etc.) on multi-vectors (a vector that is stored
    // in different buffers inside the mechanical objects)
    sofa::simulation::common::VectorOperations vop(params, this->getContext());
    sofa::simulation::common::MechanicalOperations mop(params, this->getContext());
    mop->setImplicit(true);

    core::behavior::MultiVecCoord x(&vop, xResult);

    core::behavior::MultiVecDeriv force(&vop, sofa::core::vec_id::write_access::force);

    core::behavior::MultiVecDeriv dx(&vop, core::vec_id::write_access::dx);
    dx.realloc(&vop, true, true);

    SCOPED_TIMER("StaticSolver::Solve");

    StaticResidualFunction staticResidualFunction(mop, x, force, dx, l_linearSolver.get());
    l_newtonSolver->solve(staticResidualFunction);
}

}  // namespace sofa::component::odesolver::backward
