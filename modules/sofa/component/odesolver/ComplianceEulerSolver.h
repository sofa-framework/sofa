#ifndef SOFA_COMPONENT_ODESOLVER_EULERSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_EULERSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

/** The simplest time integration.
the symplectic variant of Euler's method is applied
*/
class ComplianceEulerSolver : public core::componentmodel::behavior::OdeSolver
{
public:
    ComplianceEulerSolver();
    void solve (double dt);
    DataField<bool> firstCallToSolve;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
