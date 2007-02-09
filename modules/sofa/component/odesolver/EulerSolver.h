#ifndef SOFA_COMPONENT_ODESOLVER_EULERSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_EULERSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

class EulerSolver : public core::componentmodel::behavior::OdeSolver
{
public:
    //virtual const char* getTypeName() const { return "EulerSolver"; }
    void solve (double dt);
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
