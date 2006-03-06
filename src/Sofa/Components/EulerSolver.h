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
    void solve (double dt);
};

} // namespace Components

} // namespace Sofa

#endif
