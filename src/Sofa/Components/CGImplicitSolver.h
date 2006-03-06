#ifndef SOFA_COMPONENTS_CGIMPLICITSOLVER_H
#define SOFA_COMPONENTS_CGIMPLICITSOLVER_H

#include <Sofa/Core/OdeSolver.h>

namespace Sofa
{

namespace Components
{

class CGImplicitSolver : public Core::OdeSolver
{
public:
    unsigned int maxCGIter;
    double smallDenominatorThreshold;
    double rayleighStiffness;

    CGImplicitSolver();
    void solve (double dt);
};

} // namespace Components

} // namespace Sofa

#endif
