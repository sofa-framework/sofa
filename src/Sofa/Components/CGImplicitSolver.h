// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
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

    CGImplicitSolver();
    void solve (double dt);
    CGImplicitSolver* setMaxIter( int maxiter );

    unsigned int maxCGIter;
    double smallDenominatorThreshold;
    double tolerance;
    double rayleighStiffness;
    double rayleighDamping;
};

} // namespace Components

} // namespace Sofa

#endif
