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
#include <cstdlib>
#include <cmath>
#include <sofa/core/objectmodel/BaseNode.h>


namespace sofa::core::behavior
{

IntegrationScheme::IntegrationScheme()
    : d_rayleighStiffness(initData(&d_rayleighStiffness, (SReal)0.0, "rayleighStiffness", "Rayleigh damping coefficient related to stiffness, > 0") )
    , d_rayleighMass(initData(&d_rayleighMass, (SReal)0.0, "rayleighMass", "Rayleigh damping coefficient related to mass, > 0"))
{}

IntegrationScheme::~IntegrationScheme()
{}

void IntegrationScheme::setupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    m_params = params;
    m_dt = dt;
    m_xResult = xResult;
    m_vResult = vResult;

    m_x0.resize(getIntegrationSchemeOrder());
    m_v0.resize(getIntegrationSchemeOrder());
    m_a0.resize(getIntegrationSchemeOrder());

    doSetupIntegrationStep(params, dt, xResult, vResult);


}


bool IntegrationScheme::insertInNode( objectmodel::BaseNode* node )
{
    node->addIntegrationScheme(this);
    Inherit1::insertInNode(node);
    return true;
}

bool IntegrationScheme::removeInNode( objectmodel::BaseNode* node )
{
    node->removeIntegrationScheme(this);
    Inherit1::removeInNode(node);
    return true;
}


} // namespace sofa::core::behavior





