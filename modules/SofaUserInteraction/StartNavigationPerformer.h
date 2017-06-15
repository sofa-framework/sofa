/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_COLLISION_STARTNAVIGATIONPERFORMER_H
#define SOFA_COMPONENT_COLLISION_STARTNAVIGATIONPERFORMER_H
#include "config.h"

#include <SofaUserInteraction/MouseInteractor.h>
#include <SofaUserInteraction/InteractionPerformer.h>
#include <SofaBaseCollision/BaseContactMapper.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <SofaDeformable/SpringForceField.h>
#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaGraphComponent/AddRecordedCameraButtonSetting.h>
#include <sofa/core/visual/DisplayFlags.h>

namespace sofa
{
namespace component
{

namespace collision
{

    class SOFA_USER_INTERACTION_API StartNavigationPerformer: public InteractionPerformer
    {
    public:
        StartNavigationPerformer(BaseMouseInteractor *i)
            : InteractionPerformer(i) {};

        ~StartNavigationPerformer(){};

        // Save the current camera's position and orientation in the appropriated Data of Recorded Camera for navigation. 
        void start();
        void execute(){};

    };

}
}
}

#endif
