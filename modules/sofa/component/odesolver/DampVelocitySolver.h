#ifndef SOFA_COMPONENT_ODESOLVER_DampVelocitySolver_H
#define SOFA_COMPONENT_ODESOLVER_DampVelocitySolver_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/common/OdeSolverImpl.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

/** Velocity damping and thresholding.
This is not an ODE solver, but it can be used as a post-process after a real ODE solver.
*/
class DampVelocitySolver : public sofa::simulation::OdeSolverImpl
{
public:
    DampVelocitySolver();
    void solve (double dt);
    Data<double> rate;
    Data<double> threshold;

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    virtual double getIntegrationFactor(int inputDerivative, int outputDerivative) const
    {
        const double dt = getContext()->getDt();
        double matrix[3][3] =
        {
            { 1, 0, 0},
            { 0, exp(-rate.getValue()*dt), 0},
            { 0, 0, 0}
        };
        if (inputDerivative >= 3 || outputDerivative >= 3)
            return 0;
        else
            return matrix[outputDerivative][inputDerivative];
    }

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    ///
    virtual double getSolutionIntegrationFactor(int /*outputDerivative*/) const
    {
        return 0;
    }
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
