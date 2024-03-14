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
#define SOFA_COMPONENT_COLLISION_STARTNAVIGATIONPERFORMER_CPP

#include <sofa/gui/component/performer/StartNavigationPerformer.h>
#include <sofa/component/visual/RecordedCamera.h>
#include <sofa/helper/Factory.inl>
#include <sofa/helper/cast.h>
#include <sofa/simulation/Node.h>

using namespace sofa::core::objectmodel;

namespace sofa::gui::component::performer
{
    helper::Creator<InteractionPerformer::InteractionPerformerFactory, StartNavigationPerformer> StartNavigationPerformerClass("StartNavigation");

    void StartNavigationPerformer::start()
    {
        const sofa::simulation::Node::SPtr root = down_cast<sofa::simulation::Node>( interactor->getContext()->getRootContext() );
        if(root)
        {
            sofa::component::visual::RecordedCamera* currentCamera = root->getNodeObject<sofa::component::visual::RecordedCamera>();

            if(currentCamera)
            {
                // The navigation mode of Recorded Camera is set to true
                currentCamera->m_navigationMode.setValue(!currentCamera->m_navigationMode.getValue());
            }
        }
    }

} // namespace sofa::gui::component::performer
