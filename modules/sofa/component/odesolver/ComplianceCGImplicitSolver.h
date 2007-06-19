// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution

#ifndef SOFA_COMPONENT_ODESOLVER_COMPLIANCECGIMPLICITSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_COMPLIANCECGIMPLICITSOLVER_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/simulation/tree/OdeSolverImpl.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;

class ComplianceCGImplicitSolver : public sofa::simulation::tree::OdeSolverImpl
{
public:
    ComplianceCGImplicitSolver();

    void solve (double dt);

    DataField<unsigned> f_maxIter;
    DataField<double> f_tolerance;
    DataField<double> f_smallDenominatorThreshold;
    DataField<double> f_rayleighStiffness;
    DataField<double> f_rayleighMass;
    DataField<double> f_velocityDamping;

protected:
    bool firstCallToSolve;
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
