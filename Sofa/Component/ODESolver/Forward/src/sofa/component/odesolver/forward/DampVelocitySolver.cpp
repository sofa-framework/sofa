/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/odesolver/forward/DampVelocitySolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::odesolver::forward
{

using namespace sofa::defaulttype;
using namespace core::behavior;

void registerDampVelocitySolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Reduce the velocities.")
        .add< DampVelocitySolver >());
}

DampVelocitySolver::DampVelocitySolver()
    : d_rate(initData(&d_rate, 0.99_sreal, "rate", "Factor used to reduce the velocities. Typically between 0 and 1.") )
    , d_threshold(initData(&d_threshold, 0.0_sreal, "threshold", "Threshold under which the velocities are canceled.") )
{
    rate.setOriginalData(&d_rate);
    threshold.setOriginalData(&d_threshold);
}

void DampVelocitySolver::solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId /*xResult*/, sofa::core::MultiVecDerivId vResult)
{
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    MultiVecDeriv vel(&vop, vResult /*core::vec_id::write_access::velocity*/ );

    msg_info() <<"DampVelocitySolver, dt = "<< dt
               <<"DampVelocitySolver, initial v = "<< vel ;


    vel.teq( exp(-d_rate.getValue() * dt) );
    if(d_threshold.getValue() != 0.0 )
        vel.threshold(d_threshold.getValue() );

    msg_info() <<"DampVelocitySolver, final v = "<< vel ;
}

} // namespace sofa::component::odesolver::forward
