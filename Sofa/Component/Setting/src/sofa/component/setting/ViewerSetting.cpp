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

void registerViewerSetting(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Configuration for the Viewer of your application.")
        .add< ViewerSetting >());
}

ViewerSetting::ViewerSetting()
    : d_resolution(initData(&d_resolution, Vec<2,int>(800, 600), "resolution", "resolution of the Viewer"))
    , d_fullscreen(initData(&d_fullscreen, false, "fullscreen", "Fullscreen mode"))
    , d_cameraMode(initData(&d_cameraMode, {"Perspective", "Orthographic"}, "cameraMode", "Camera mode"))
    , d_objectPickingMethod(initData(&d_objectPickingMethod, {"Ray casting", "Selection buffer"}, "objectPickingMethod", "The method used to pick objects"))
    , d_showSelectedNodeBoundingBox(initData(&d_showSelectedNodeBoundingBox, true, "showSelectedNodeBoundingBox", "Show the bounding box of selected nodes"))
    , d_showSelectedObjectBoundingBox(initData(&d_showSelectedObjectBoundingBox, true, "showSelectedObjectBoundingBox", "Show the bounding box when components selected"))
    , d_showSelectedObjectPositions(initData(&d_showSelectedObjectPositions, true, "showSelectedObjectPositions", "Show the positions when a components with 'position' are selected"))
    , d_showSelectedObjectSurfaces(initData(&d_showSelectedObjectSurfaces, true, "showSelectedObjectSurfaces", "Show the surfaces when components with surface topology are selected"))
    , d_showSelectedObjectVolumes(initData(&d_showSelectedObjectVolumes, true, "showSelectedObjectVolumes", "Show the volumes when components with volume topology are selected"))
    , d_showSelectedObjectIndices(initData(&d_showSelectedObjectIndices, true, "showSelectedObjectIndices", "Show the position's indices for components with positions are selected"))
    , d_selectedVisualScaling(initData(&d_selectedVisualScaling, 0.02, "showSelectedVisualScaling", "Scale factor for the rendering of selected object"))
{
    d_resolution.setGroup("Viewport");
    d_fullscreen.setGroup("Viewport");

    d_cameraMode.setGroup("Camera");

    d_objectPickingMethod.setGroup("Selection");
    d_showSelectedNodeBoundingBox.setGroup("Selection");
    d_showSelectedObjectBoundingBox.setGroup("Selection");
    d_showSelectedObjectPositions.setGroup("Selection");
    d_showSelectedObjectSurfaces.setGroup("Selection");
    d_showSelectedObjectVolumes.setGroup("Selection");
    d_showSelectedObjectIndices.setGroup("Selection");
    d_selectedVisualScaling.setGroup("Selection");
}

} // namespace sofa::component::setting
