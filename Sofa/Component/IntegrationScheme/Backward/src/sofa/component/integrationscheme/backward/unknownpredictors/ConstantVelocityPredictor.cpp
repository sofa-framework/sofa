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
#include <sofa/component/integrationscheme/backward/unknownpredictors/ConstantVelocityPredictor.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/VectorOperations.h>

namespace sofa::component::integrationscheme::backward::unknownpredictors
{


void ConstantVelocityPredictor::setUnknown(simulation::integrationscheme::ImplicitIntegrationScheme * integrationScheme)
{
    switch (integrationScheme->getUnknownDerivativeOrder())
    {
        case 1:
            integrationScheme->m_vop->v_clear(integrationScheme->m_systemUnknown);
            break;

        case 2:
            integrationScheme->m_vop->v_eq(integrationScheme->m_systemUnknown,integrationScheme->m_acceleration);
            break;

        default:
            msg_warning()<<"This integration scheme type is not supported." ;
            break;
    }
}


void registerConstantVelocityPredictor(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("TODO")
        .add< ConstantVelocityPredictor >());
}

}
