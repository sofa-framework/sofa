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
#define SOFA_COMPONENT_COLLISION_ADDRECORDEDCAMERAPERFORMER_CPP

#include <SofaUserInteraction/AddRecordedCameraPerformer.h>
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
            helper::Creator<InteractionPerformer::InteractionPerformerFactory, AddRecordedCameraPerformer> AddRecordedCameraPerformerClass("AddRecordedCamera");

            void AddRecordedCameraPerformer::start()
            {
                sofa::simulation::Node::SPtr root = down_cast<sofa::simulation::Node>( interactor->getContext()->getRootContext() );
                if(root)
                {
                    sofa::component::visualmodel::RecordedCamera* currentCamera = root->getNodeObject<sofa::component::visualmodel::RecordedCamera>();

                    if(currentCamera)
                    {
                        // Set the current camera's position in recorded camera for navigation
                        sofa::component::visualmodel::RecordedCamera::Vec3 _pos = currentCamera->p_position.getValue();
                        sofa::helper::vector<sofa::component::visualmodel::RecordedCamera::Vec3> posis = currentCamera->m_translationPositions.getValue();
                        posis.push_back(_pos);
                        currentCamera->m_translationPositions.setValue(posis);

                        // Set the current camera's orientation in recorded camera for navigation
                        sofa::component::visualmodel::RecordedCamera::Quat _ori = currentCamera->p_orientation.getValue();
                        sofa::helper::vector<sofa::component::visualmodel::RecordedCamera::Quat>oris = currentCamera->m_translationOrientations.getValue();//push_back(m_vectorOrientations);
                        oris.push_back(_ori);
                        currentCamera->m_translationOrientations.setValue(oris);

                    }
                }
            }

        }// namespace collision
    }// namespace component
}// namespace sofa
