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
#include <sofa/core/behavior/IntegrationScheme.h>
#include <sofa/simulation/integrationschemes/ImplicitIntegrationScheme.h>
#include <cstdlib>
#include <cmath>
#include <sofa/core/objectmodel/BaseNode.h>

#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>

namespace sofa::simulation::integrationschemes
{


ImplicitIntegrationScheme::ImplicitIntegrationScheme()
: d_rayleighStiffness(initData(&d_rayleighStiffness, 0.0_sreal, "rayleighStiffness", "Rayleigh damping coefficient related to stiffness, > 0") )
, d_rayleighMass(initData(&d_rayleighMass, 0.0_sreal, "rayleighMass", "Rayleigh damping coefficient related to mass, > 0"))
, m_vop(nullptr)
, m_mop(nullptr)
{

}

void ImplicitIntegrationScheme::setupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{

    m_vop = std::make_shared<sofa::simulation::common::VectorOperations>( params, this->getContext() );
    m_mop = std::make_unique<sofa::simulation::common::MechanicalOperations>( params, this->getContext() );

    m_params = params;
    m_dt = dt;
    m_xResult = xResult;
    m_vResult = vResult;


    doSetupIntegrationStep(params, dt, xResult, vResult);


}

void ImplicitIntegrationScheme::integrate(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    setupIntegrationStep(params, dt, xResult, vResult);
    computeRHS(true);
    computeLHS(true);
    solveLinearEquation();
    updateStatesFromLinearSolution(1.0, true);
}

}