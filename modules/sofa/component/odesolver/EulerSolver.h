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
Two variants are available, depending on the value of field "symplectic".
If true (the default), the symplectic variant of Euler's method is applied:
If false, the basic Euler's method is applied (less robust)
*/
class EulerSolver : public sofa::simulation::OdeSolverImpl
{
public:
    EulerSolver();
    void solve (double dt);
    Data<bool> symplectic;

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    virtual double getIntegrationFactor(int inputDerivative, int outputDerivative) const
    {
        const double dt = getContext()->getDt();
        double matrix[3][3] =
        {
            { 1, dt, ((symplectic.getValue())?dt*dt:0.0)},
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
        double vect[3] = { ((symplectic.getValue())?dt*dt:0.0), dt, 1};
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
