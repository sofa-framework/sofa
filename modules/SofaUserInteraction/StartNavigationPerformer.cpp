/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaUserInteraction/StartNavigationPerformer.h>
#include <SofaGeneralVisual/RecordedCamera.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/Factory.inl>
#include <SofaRigid/JointSpringForceField.inl>
#include <SofaDeformable/SpringForceField.inl>
#include <SofaDeformable/StiffSpringForceField.inl>
#include <sofa/helper/cast.h>


using namespace sofa::component::interactionforcefield;
using namespace sofa::core::objectmodel;
namespace sofa
{

    namespace component
    {

        namespace collision
        {
            helper::Creator<InteractionPerformer::InteractionPerformerFactory, StartNavigationPerformer> StartNavigationPerformerClass("StartNavigation");

            void StartNavigationPerformer::start()
            {
                sofa::simulation::Node::SPtr root = down_cast<sofa::simulation::Node>( interactor->getContext()->getRootContext() );
                if(root)
                {
                    sofa::component::visualmodel::RecordedCamera* currentCamera = root->getNodeObject<sofa::component::visualmodel::RecordedCamera>();

                    if(currentCamera)
                    {
                        // The navigation mode of Recorded Camera is set to true
                        currentCamera->m_navigationMode.setValue(!currentCamera->m_navigationMode.getValue());
                    }
                }
            }

        }// namespace collision
    }// namespace component
}// namespace sofa
