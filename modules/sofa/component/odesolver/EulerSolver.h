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
Two variants are available, depending on the value of field "symplectic".
If true (the default), the symplectic variant of Euler's method is applied:
If false, the basic Euler's method is applied (less robust)
*/
class EulerSolver : public sofa::simulation::tree::OdeSolverImpl
{
public:
    EulerSolver();
    void solve (double dt);
    DataField<bool> symplectic;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
