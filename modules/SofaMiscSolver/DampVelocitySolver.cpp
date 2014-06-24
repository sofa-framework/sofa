/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaMiscSolver/DampVelocitySolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/VectorOperations.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>




namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace core::behavior;

int DampVelocitySolverClass = core::RegisterObject("Reduce the velocities")
        .add< DampVelocitySolver >()
        .addAlias("DampVelocity")
        ;

SOFA_DECL_CLASS(DampVelocity);

DampVelocitySolver::DampVelocitySolver()
    : rate( initData( &rate, 0.99, "rate", "Factor used to reduce the velocities. Typically between 0 and 1.") )
    , threshold( initData( &threshold, 0.0, "threshold", "Threshold under which the velocities are canceled.") )
{}

void DampVelocitySolver::solve(const core::ExecParams* params /* PARAMS FIRST */, double dt, sofa::core::MultiVecCoordId /*xResult*/, sofa::core::MultiVecDerivId vResult)
{
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    //sofa::simulation::common::MechanicalOperations mop( this->getContext() );
    MultiVecDeriv vel(&vop, vResult /*core::VecDerivId::velocity()*/ );
    bool printLog = f_printLog.getValue();

    if( printLog )
    {
        serr<<"DampVelocitySolver, dt = "<< dt <<sendl;
        serr<<"DampVelocitySolver, initial v = "<< vel <<sendl;
    }

    //mop.addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    vel.teq( exp(-rate.getValue()*dt) );
    if( threshold.getValue() != 0.0 )
        vel.threshold( threshold.getValue() );

    if( printLog )
    {
        serr<<"DampVelocitySolver, final v = "<< vel <<sendl;
    }
}

} // namespace odesolver

} // namespace component

} // namespace sofa

