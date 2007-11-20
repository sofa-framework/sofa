#ifndef SOFA_COMPONENT_ODESOLVER_EULERSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_EULERSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/tree/OdeSolverImpl.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

/** The simplest time integration.
the symplectic variant of Euler's method is applied
*/
class ComplianceEulerSolver : public sofa::simulation::tree::OdeSolverImpl
{
public:
    ComplianceEulerSolver();
    void solve (double dt);
    Data<bool> firstCallToSolve;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
