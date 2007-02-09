// Author: Jeremie Allard, Sim Group @ CIMIT, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_ODESOLVER_BICGSTABIMPLICITSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_BICGSTABIMPLICITSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>

namespace sofa
{

namespace component
{

namespace odesolver
{
/** Implicit integration solver able to handle degenerate equation systems.
*/
class BiCGStabImplicitSolver : public core::componentmodel::behavior::OdeSolver
{
public:
    typedef core::componentmodel::behavior::OdeSolver Inherited;

    BiCGStabImplicitSolver();
    virtual void parse(core::objectmodel::BaseObjectDescription* arg);
    void solve (double dt);
    BiCGStabImplicitSolver* setMaxIter( int maxiter );

    unsigned int maxCGIter;
    double smallDenominatorThreshold;
    double tolerance;
    double rayleighStiffness;

    bool getDebug()
    {
        return false;
    }
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif


