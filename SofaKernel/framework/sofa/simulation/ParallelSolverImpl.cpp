/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "ParallelSolverImpl.h"
#include "ParallelMechanicalVisitor.h"
#include <sofa/simulation/MechanicalMatrixVisitor.h>
#include <sofa/simulation/MechanicalVPrintVisitor.h>
#include <sofa/simulation/VelocityThresholdVisitor.h>
#include <sofa/core/behavior/LinearSolver.h>


#include <stdlib.h>
#include <math.h>

namespace sofa
{

namespace simulation
{
namespace common
{


ParallelSolverImpl::ParallelSolverImpl()
{}

ParallelSolverImpl::~ParallelSolverImpl()
{}



void ParallelSolverImpl::v_op(VecId v, VecId a, VecId b, Shared<double>  &f) ///< v=a+b*f
{
    ParallelMechanicalVOpVisitor(v,a,b,1.0,&f).execute( getContext() );
}
void ParallelSolverImpl::v_peq(VecId v, VecId a, Shared<double> &fSh,double f) ///< v+=f*a
{
    ParallelMechanicalVOpVisitor(v,v,a,f,&fSh).execute( getContext() );
}
void ParallelSolverImpl::v_peq(VecId v, VecId a, double f) ///< v+=f*a
{
    ParallelMechanicalVOpVisitor(v,v,a,f).execute( getContext() );
}
void ParallelSolverImpl::v_meq(VecId v, VecId a, Shared<double> &fSh) ///< v+=f*a
{
    ParallelMechanicalVOpMecVisitor(v,a,&fSh).execute( getContext() );
}

void ParallelSolverImpl::v_dot(Shared<double> &result,VecId a, VecId b) ///< a dot b ( get result using finish )
{

    ParallelMechanicalVDotVisitor(&result,a,b).execute( getContext() );
}



} // namespace simulation
} // namespace simulation

} // namespace sofa
