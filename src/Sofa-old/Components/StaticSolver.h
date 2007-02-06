// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENTS_STATICSOLVER_H
#define SOFA_COMPONENTS_STATICSOLVER_H

#include <Sofa-old/Core/OdeSolver.h>


namespace Sofa
{

namespace Components
{
using Common::DataField;

class StaticSolver : public Core::OdeSolver
{

public:
    DataField<unsigned int> f_maxCGIter;

    StaticSolver();

    double smallDenominatorThreshold;
    virtual const char* getTypeName() const { return "StaticSolver"; }


    void solve (double dt);
};

} // namespace Components

} // namespace Sofa

#endif
