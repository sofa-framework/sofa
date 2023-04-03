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

#include <sofa/component/setting/ViewerSetting.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::setting
{

using namespace sofa::type;
using namespace sofa::helper;

int ViewerSettingClass = core::RegisterObject("Configuration for the Viewer of your application")
        .add< ViewerSetting >()
        .addAlias("Viewer")
        ;

ViewerSetting::ViewerSetting()
    : resolution(initData(&resolution, Vec<2,int>(800,600), "resolution", "resolution of the Viewer"))
    , fullscreen(initData(&fullscreen, false, "fullscreen", "Fullscreen mode"))
    , cameraMode(initData(&cameraMode, {"Perspective","Orthographic"}, "cameraMode", "Camera mode"))
    , objectPickingMethod(initData(&objectPickingMethod, {"Ray casting","Selection buffer"}, "objectPickingMethod","The method used to pick objects"))
{}

} // namespace sofa::component::setting
