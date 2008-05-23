#ifndef SOFA_COMPONENT_ODESOLVER_EULERSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_EULERSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/common/OdeSolverImpl.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

/** The simplest time integration.
the symplectic variant of Euler's method is applied
*/
class ComplianceEulerSolver : public sofa::simulation::OdeSolverImpl
{
public:
    ComplianceEulerSolver();
    void solve (double dt);
    Data<bool> firstCallToSolve;

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    virtual double getIntegrationFactor(int inputDerivative, int outputDerivative) const
    {
        const double dt = getContext()->getDt();
        double matrix[3][3] =
        {
            { 1, dt, dt*dt},
            { 0, 1, dt},
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
    virtual double getSolutionIntegrationFactor(int outputDerivative) const
    {
        const double dt = getContext()->getDt();
        double vect[3] = { dt*dt, dt, 1};
        if (outputDerivative >= 3)
            return 0;
        else
            return vect[outputDerivative];
    }
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
