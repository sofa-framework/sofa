#ifndef SOFA_COMPONENTS_STATICSOLVER_H
#define SOFA_COMPONENTS_STATICSOLVER_H

#include <Sofa/Core/OdeSolver.h>

namespace Sofa
{

namespace Components
{

class StaticSolver : public Core::OdeSolver
{
public:
    StaticSolver();

    unsigned int maxCGIter;
    double smallDenominatorThreshold;

    void solve (double dt);
};

} // namespace Components

} // namespace Sofa

#endif
