#ifndef SOFA_COMPONENTS_RUNGEKUTTA4SOLVER_H
#define SOFA_COMPONENTS_RUNGEKUTTA4SOLVER_H

#include <Sofa-old/Core/OdeSolver.h>

namespace Sofa
{

namespace Components
{

class RungeKutta4Solver : public Core::OdeSolver
{
public:
    void solve (double dt);
};

} // namespace Components

} // namespace Sofa

#endif
