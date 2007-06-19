#ifndef SOFA_COMPONENT_ODESOLVER_EULERKAAPISOLVER_H
#define SOFA_COMPONENT_ODESOLVER_EULERKAAPISOLVER_H

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
class EulerKaapiSolver : public sofa::simulation::tree::OdeSolverImpl
{
public:
    EulerKaapiSolver();
    void solve (double dt);
    void computeAcc (double t, VecId a, VecId x, VecId v);
    DataField<bool> symplectic;
    void v_free(VecId v);
    void v_clear(VecId v); ///< v=0
    VecId v_alloc(VecId::Type t);

    void propagatePositionAndVelocity(double t, VecId x, VecId v);
    void computeForce(VecId result);
    void accFromF(VecId a, VecId f);
    void projectResponse(VecId dx, double **W=NULL);
    void v_peq(VecId v, VecId a, double f); ///< v+=f*a


};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
