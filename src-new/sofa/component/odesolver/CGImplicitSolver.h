// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENTS_CGIMPLICITSOLVER_H
#define SOFA_COMPONENTS_CGIMPLICITSOLVER_H

#include <Sofa-old/Core/OdeSolver.h>

namespace Sofa
{

namespace Components
{
using namespace Common;

class CGImplicitSolver : public Core::OdeSolver
{
public:

    CGImplicitSolver();
    virtual const char* getTypeName() const { return "CGImplicit"; }

    void solve (double dt);
    //CGImplicitSolver* setMaxIter( int maxiter );

    DataField<unsigned> f_maxIter;
    DataField<double> f_tolerance;
    DataField<double> f_smallDenominatorThreshold;
    DataField<double> f_rayleighStiffness;
    DataField<double> f_rayleighMass;
    DataField<double> f_velocityDamping;

protected:

    /*	unsigned maxCGIter;
    	double smallDenominatorThreshold;
    	double tolerance;
    	double rayleighStiffness;
    	double rayleighMass;
    	double velocityDamping;*/
};

} // namespace Components

} // namespace Sofa

#endif
