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
#define SOFA_COMPONENT_COLLISION_ADDRECORDEDCAMERAPERFORMER_CPP

#include <sofa/gui/component/performer/AddRecordedCameraPerformer.h>

#include <sofa/helper/cast.h>
#include <sofa/helper/Factory.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/visual/RecordedCamera.h>
#include <sofa/gui/component/performer/MouseInteractor.h>


using namespace sofa::core::objectmodel;

namespace sofa::gui::component::performer
{

void AddRecordedCameraPerformer::start()
{
    const sofa::simulation::Node::SPtr root = down_cast<sofa::simulation::Node>( interactor->getContext()->getRootContext() );
    if(root)
    {
        sofa::component::visual::RecordedCamera* currentCamera = root->getNodeObject<sofa::component::visual::RecordedCamera>();

        if(currentCamera)
        {
            // Set the current camera's position in recorded camera for navigation
            const type::Vec3 _pos = currentCamera->p_position.getValue();
            sofa::type::vector<type::Vec3> posis = currentCamera->m_translationPositions.getValue();
            posis.push_back(_pos);
            currentCamera->m_translationPositions.setValue(posis);

            // Set the current camera's orientation in recorded camera for navigation
            const sofa::component::visual::RecordedCamera::Quat _ori = currentCamera->p_orientation.getValue();
            sofa::type::vector<sofa::component::visual::RecordedCamera::Quat>oris = currentCamera->m_translationOrientations.getValue();//push_back(m_vectorOrientations);
            oris.push_back(_ori);
            currentCamera->m_translationOrientations.setValue(oris);

        }
    }
}


helper::Creator<InteractionPerformer::InteractionPerformerFactory, AddRecordedCameraPerformer> AddRecordedCameraPerformerClass("AddRecordedCamera");

}// namespace sofa::gui::component::performer
