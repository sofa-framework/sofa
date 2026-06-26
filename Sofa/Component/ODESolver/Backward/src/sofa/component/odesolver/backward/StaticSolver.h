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
#include <optional>

namespace sofa::component::odesolver::backward
{

class SOFA_COMPONENT_ODESOLVER_BACKWARD_API StaticSolver :
    public core::behavior::OdeSolver,
    public core::behavior::LinearSolverAccessor
{
public:
    SOFA_CLASS2(StaticSolver, core::behavior::OdeSolver, core::behavior::LinearSolverAccessor);
    StaticSolver();

    void solve(
        const core::ExecParams* params,
        SReal dt,
        core::MultiVecCoordId xResult,
        core::MultiVecDerivId vResult) override;

    void parse(core::objectmodel::BaseObjectDescription* arg) override;
    void init() override;

    SingleLink<StaticSolver, NewtonRaphsonSolver,
               BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK>
        l_newtonSolver;

    auto squared_residual_norms() const -> const std::vector<SReal> & = delete;
    auto squared_increment_norms() const -> const std::vector<SReal> & = delete;


    /// Given a displacement as computed by the linear system inversion, how much will it affect the velocity
    ///
    /// This method is used to compute the compliance for contact corrections
    /// For Euler methods, it is typically dt.
    SReal getVelocityIntegrationFactor() const override
    {
        return 1.0; // getContext()->getDt();
    }

    /// Given a displacement as computed by the linear system inversion, how much will it affect the position
    ///
    /// This method is used to compute the compliance for contact corrections
    /// For Euler methods, it is typically dt².
    SReal getPositionIntegrationFactor() const override
    {
        return getPositionIntegrationFactor(getContext()->getDt());
    }

    virtual SReal getPositionIntegrationFactor(SReal dt ) const
    {
        return dt;
    }

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    ///
    /// This method is used to compute the compliance for contact corrections.
    /// For example, a backward-Euler dynamic implicit integrator would use:
    /// Input:      x_t  v_t  a_{t+dt}
    /// x_{t+dt}     1    dt  dt^2
    /// v_{t+dt}     0    1   dt
    ///
    /// If the linear system is expressed on s = a_{t+dt} dt, then the final factors are:
    /// Input:      x_t   v_t    a_t  s
    /// x_{t+dt}     1    dt     0    dt
    /// v_{t+dt}     0    1      0    1
    /// a_{t+dt}     0    0      0    1/dt
    /// The last column is returned by the getSolutionIntegrationFactor method.
    SReal getIntegrationFactor(int inputDerivative, int outputDerivative) const override
    {
        return getIntegrationFactor(inputDerivative, outputDerivative, getContext()->getDt());
    }

    SReal getIntegrationFactor(int inputDerivative, int outputDerivative, SReal dt) const
    {
        const SReal matrix[3][3] =
            {
                { 1, dt, 0},
                { 0, 1, 0},
                { 0, 0, 0}
            };
        if (inputDerivative >= 3 || outputDerivative >= 3)
            return 0;
        else
            return matrix[outputDerivative][inputDerivative];
    }

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    SReal getSolutionIntegrationFactor(int outputDerivative) const override
    {
        return getSolutionIntegrationFactor(outputDerivative, getContext()->getDt());
    }

    SReal getSolutionIntegrationFactor(int outputDerivative, SReal dt) const
    {
        const SReal vect[3] = { dt, 1, 1/dt};
        if (outputDerivative >= 3)
            return 0;
        else
            return vect[outputDerivative];
    }


protected:

    template<class T>
    struct NewtonRaphsonDeprecatedData : core::objectmodel::lifecycle::RemovedData
    {
        NewtonRaphsonDeprecatedData(Base* b, const std::string name)
            : RemovedData(b, "v25.06", "v25.12", name, "The Data related to the Newton-Raphson parameters must be defined in the NewtonRaphsonSolver component.")
        {}

        std::optional<T> value;
    };

    SOFA_ATTRIBUTE_DISABLED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData<int> d_newton_iterations;
    SOFA_ATTRIBUTE_DISABLED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData<SReal> d_absolute_correction_tolerance_threshold;
    SOFA_ATTRIBUTE_DISABLED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData<SReal> d_relative_correction_tolerance_threshold;
    SOFA_ATTRIBUTE_DISABLED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData<SReal> d_absolute_residual_tolerance_threshold;
    SOFA_ATTRIBUTE_DISABLED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData<SReal> d_relative_residual_tolerance_threshold;
    SOFA_ATTRIBUTE_DISABLED__NEWTONRAPHSON_IN_STATICSOLVER() NewtonRaphsonDeprecatedData<SReal> d_should_diverge_when_residual_is_growing;
};

}
