// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution

#ifndef SOFA_COMPONENT_ODESOLVER_COMPLIANCECGIMPLICITSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_COMPLIANCECGIMPLICITSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/common/OdeSolverImpl.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;

class ComplianceCGImplicitSolver : public sofa::simulation::OdeSolverImpl
{
public:
    ComplianceCGImplicitSolver();

    void solve (double dt);

    Data<unsigned> f_maxIter;
    Data<double> f_tolerance;
    Data<double> f_smallDenominatorThreshold;
    Data<double> f_rayleighStiffness;
    Data<double> f_rayleighMass;
    Data<double> f_velocityDamping;

protected:
    bool firstCallToSolve;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
