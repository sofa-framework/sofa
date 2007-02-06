// Author: Jeremie Allard, Sim Group @ CIMIT, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENTS_BICGSTABIMPLICITSOLVER_H
#define SOFA_COMPONENTS_BICGSTABIMPLICITSOLVER_H

#include <Sofa/Core/OdeSolver.h>

namespace Sofa
{

namespace Components
{

class BiCGStabImplicitSolver : public Core::OdeSolver
{
public:

    BiCGStabImplicitSolver();
    void solve (double dt);
    BiCGStabImplicitSolver* setMaxIter( int maxiter );

    unsigned int maxCGIter;
    double smallDenominatorThreshold;
    double tolerance;
    double rayleighStiffness;

    bool getDebug() { return false; }
};

} // namespace Components

} // namespace Sofa

#endif
