#ifndef SOFA_COMPONENTS_EULERSOLVER_H
#define SOFA_COMPONENTS_EULERSOLVER_H

#include <Sofa/Core/OdeSolver.h>

namespace Sofa
{

namespace Components
{

class EulerSolver : public Core::OdeSolver
{
public:
    virtual const char* getTypeName() const { return "EulerSolver"; }
    void solve (double dt);
};

} // namespace Components

} // namespace Sofa

#endif
