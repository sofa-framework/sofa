#ifndef SOFA_COMPONENT_ODESOLVER_DampVelocitySolver_H
#define SOFA_COMPONENT_ODESOLVER_DampVelocitySolver_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/tree/OdeSolverImpl.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

/** Velocity damping and thresholding.
This is not an ODE solver, but it can be used as a post-process after a real ODE solver.
*/
class DampVelocitySolver : public sofa::simulation::tree::OdeSolverImpl
{
public:
    DampVelocitySolver();
    void solve (double dt);
    DataField<double> rate;
    DataField<double> threshold;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
