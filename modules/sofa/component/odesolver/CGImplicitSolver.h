// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_ODESOLVER_CGIMPLICITSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_CGIMPLICITSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;

class CGImplicitSolver : public core::componentmodel::behavior::OdeSolver
{
public:

    CGImplicitSolver();
    //virtual const char* getTypeName() const { return "CGImplicit"; }

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

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
