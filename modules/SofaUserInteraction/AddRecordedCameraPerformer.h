/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_ADDRECORDEDCAMERAPERFORMER_H
#define SOFA_COMPONENT_COLLISION_ADDRECORDEDCAMERAPERFORMER_H
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

    class SOFA_USER_INTERACTION_API AddRecordedCameraPerformer: public InteractionPerformer
    {
    public:
        AddRecordedCameraPerformer(BaseMouseInteractor *i)
            : InteractionPerformer(i) {};

        ~AddRecordedCameraPerformer(){};

        // Save the current camera's position and orientation in the appropriate Data of Recorded Camera for navigation. 
        void start();
        void execute(){};

    };

}
}
}

#endif
