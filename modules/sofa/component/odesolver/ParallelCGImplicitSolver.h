/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_SMP_COMPONENT_PARALLELCGIMPLICITSOLVER_H
#define SOFA_SMP_COMPONENT_PARALLELCGIMPLICITSOLVER_H

#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/component/odesolver/ParallelOdeSolverImpl.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace sofa::simulation;


/** Implicit time integrator using the filtered conjugate gradient solution [Baraff&Witkin 98].
*/
class  ParallelCGImplicitSolver:   public ParallelOdeSolverImpl
{
    typedef ParallelOdeSolverImpl::MultiVector MultiVector;
    typedef ParallelOdeSolverImpl::VecId VecId;

public:
    SOFA_CLASS(ParallelCGImplicitSolver, sofa::component::odesolver::ParallelOdeSolverImpl);


    ParallelCGImplicitSolver();
    ~ParallelCGImplicitSolver();

    void solve (double dt);
    void cgLoop( MultiVector &x, MultiVector &r,MultiVector& p,MultiVector &q,double &h,const bool verbose);
    Data<unsigned> f_maxIter;
    Data<double> f_tolerance;
    Data<double> f_smallDenominatorThreshold;
    Data<double> f_rayleighStiffness;
    Data<double> f_rayleighMass;
    Data<double> f_velocityDamping;
    Data<bool> f_verbose;

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    ///
    /// This method is used to compute the compliance for contact corrections.
    /// For example, a backward-Euler dynamic implicit integrator would use:
    /// Input:      x_t  v_t  a_{t+dt}
    /// x_{t+dt}     1    dt  dt^2
    /// v_{t+dt}     0    1   dt
    ///
    /// If the linear system is expressed on s = a_{t+dt} dt, then the final factors are:
    /// Input:      x_t   v_t    a_t  s
    /// x_{t+dt}     1    dt     0    dt
    /// v_{t+dt}     0    1      0    1
    /// a_{t+dt}     0    0      0    1/dt
    /// The last column is returned by the getSolutionIntegrationFactor method.
    double getIntegrationFactor(int inputDerivative, int outputDerivative) const
    {
        const double dt = getContext()->getDt();
        double matrix[3][3] =
        {
            { 1, dt, 0},
            { 0, 1, 0},
            { 0, 0, 0}
        };
        if (inputDerivative >= 3 || outputDerivative >= 3)
            return 0;
        else
            return matrix[outputDerivative][inputDerivative];
    }

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    double getSolutionIntegrationFactor(int outputDerivative) const
    {
        const double dt = getContext()->getDt();
        double vect[3] = { dt, 1, 1/dt};
        if (outputDerivative >= 3)
            return 0;
        else
            return vect[outputDerivative];
    }

protected:



protected:
    Shared<double> *rhoSh,*rho_1Sh,*alphaSh,*betaSh,*denSh,normbSh;
    Shared<bool> *breakCondition;

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
