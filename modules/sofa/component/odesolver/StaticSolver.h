// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_ODESOLVER_STATICSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_STATICSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>


namespace sofa
{

namespace component
{

namespace odesolver
{

using core::objectmodel::DataField;

/** Finds the static equilibrium of a system. Can diverge when there are an infinity of solutions. */
class StaticSolver : public core::componentmodel::behavior::OdeSolver
{

public:
    DataField<unsigned int> f_maxCGIter;
    DataField<double> f_smallDenominatorThreshold;

    StaticSolver();

    //virtual const char* getTypeName() const { return "StaticSolver"; }

    void solve (double dt);
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
