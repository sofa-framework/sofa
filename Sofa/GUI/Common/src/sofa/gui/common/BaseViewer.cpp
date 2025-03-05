/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/gui/common/BaseViewer.h>
#include <sofa/gui/common/PickHandler.h>
#include <sofa/gui/common/BaseGUI.h>

#include <sofa/helper/Factory.inl>
#include <sofa/core/visual/DisplayFlags.h>
#include <sofa/component/setting/ViewerSetting.h>
#include <sofa/component/visual/VisualStyle.h>
#include <sofa/component/visual/InteractiveCamera.h>

#include <sofa/core/ComponentNameHelper.h>

namespace sofa::gui::common
{

BaseViewer::BaseViewer()
    : groot(nullptr)
    , currentCamera(nullptr)
    , _video(false)
    , m_isVideoButtonPressed(false)
    , m_bShowAxis(false)
    , backgroundColour(type::Vec3())
    , backgroundImageFile("textures/SOFA_logo.bmp")
    , ambientColour(type::Vec3())
    , pick(std::make_unique<PickHandler>())
    , _screenshotDirectory(".")
{}

BaseViewer::~BaseViewer()
{

}

sofa::simulation::Node* BaseViewer::getScene()
{
    return groot.get();
}
const std::string& BaseViewer::getSceneFileName()
{
    return sceneFileName;
}
void BaseViewer::setSceneFileName(const std::string &f)
{
    sceneFileName = f;
}

void BaseViewer::setScene(sofa::simulation::Node::SPtr scene, const char* filename /* = nullptr */, bool /* = false */)
{
    std::string prefix = "";
    if (filename)
        prefix = sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(filename) + "_";
    setPrefix(prefix);

    sceneFileName = filename ? filename : std::string("default.scn");
    groot = scene;
    initTexturesDone = false;
}

void BaseViewer::setCameraMode(core::visual::VisualParams::CameraType mode)
{
    currentCamera->setCameraType(mode);
}

bool BaseViewer::ready()
{
    return true;
}

void BaseViewer::wait()
{
}

void BaseViewer::configure(sofa::component::setting::ViewerSetting* viewerConf)
{
    using namespace core::visual;
    if (viewerConf->d_cameraMode.getValue().getSelectedId() == VisualParams::ORTHOGRAPHIC_TYPE)
        setCameraMode(VisualParams::ORTHOGRAPHIC_TYPE);
    else
        setCameraMode(VisualParams::PERSPECTIVE_TYPE);
}

//Functions needed to take a screenshot
const std::string BaseViewer::screenshotName()
{
    return "";
}

void BaseViewer::setPrefix(const std::string& prefix, bool prependDirectory)
{
    SOFA_UNUSED(prefix);
    SOFA_UNUSED(prependDirectory);
}

void BaseViewer::screenshot(const std::string& filename, int compression_level)
{
    SOFA_UNUSED(filename);
    SOFA_UNUSED(compression_level);
}

void BaseViewer::getView(type::Vec3& pos, type::Quat<SReal>& ori) const
{
    if (!currentCamera)
        return;

    const type::Vec3d& camPosition = currentCamera->getPosition();
    const type::Quat<SReal>& camOrientation = currentCamera->getOrientation();

    pos[0] = camPosition[0];
    pos[1] = camPosition[1];
    pos[2] = camPosition[2];

    ori[0] = camOrientation[0];
    ori[1] = camOrientation[1];
    ori[2] = camOrientation[2];
    ori[3] = camOrientation[3];
}

void BaseViewer::setView(const type::Vec3& pos, const type::Quat<SReal> &ori)
{
    type::Vec3d position;
    type::Quat<SReal> orientation;
    for (unsigned int i=0 ; i<3 ; i++)
    {
        position[i] = pos[i];
        orientation[i] = ori[i];
    }
    orientation[3] = ori[3];

    if (currentCamera)
        currentCamera->setView(position, orientation);

    redraw();
}

void BaseViewer::moveView(const type::Vec3& pos, const type::Quat<SReal> &ori)
{
    if (!currentCamera)
        return;

    currentCamera->moveCamera(pos, ori);

    redraw();
}

void BaseViewer::newView()
{
    if (!currentCamera || !groot)
        return;

    currentCamera->setDefaultView(groot->getGravity());
}

void BaseViewer::resetView()
{
    redraw();
}

void BaseViewer::setBackgroundColour(float r, float g, float b)
{
    _background = 3;
    backgroundColour[0] = r;
    backgroundColour[1] = g;
    backgroundColour[2] = b;
}

void BaseViewer::setBackgroundImage(std::string imageFileName)
{
    SOFA_UNUSED(imageFileName);
}

std::string BaseViewer::getBackgroundImage()
{
    return backgroundImageFile;
}

PickHandler* BaseViewer::getPickHandler()
{
    return pick.get();
}

bool BaseViewer::load()
{
    if (groot)
    {
        groot->get(currentCamera);
        if (!currentCamera)
        {
            currentCamera = sofa::core::objectmodel::New<sofa::component::visual::InteractiveCamera>();
            currentCamera->setName(groot->getNameHelper().resolveName(currentCamera->getClassName(), sofa::core::ComponentNameHelper::Convention::python));
            groot->addObject(currentCamera);
            //currentCamera->d_position.forceSet();
            //currentCamera->d_orientation.forceSet();
            currentCamera->bwdInit();
        }
        sofa::component::visual::VisualStyle::SPtr visualStyle = nullptr;
        groot->get(visualStyle);
        if (!visualStyle)
        {
            visualStyle = sofa::core::objectmodel::New<sofa::component::visual::VisualStyle>();
            visualStyle->setName(groot->getNameHelper().resolveName(visualStyle->getClassName(), sofa::core::ComponentNameHelper::Convention::python));

            core::visual::DisplayFlags* displayFlags = visualStyle->d_displayFlags.beginEdit();
            displayFlags->setShowVisualModels(sofa::core::visual::tristate::true_value);
            visualStyle->d_displayFlags.endEdit();

            groot->addObject(visualStyle);
            visualStyle->init();
        }

        currentCamera->setBoundingBox(groot->f_bbox.getValue().minBBox(), groot->f_bbox.getValue().maxBBox());

        // init pickHandler
        pick->init(groot.get());

        return true;
    }

    return false;
}

bool BaseViewer::unload()
{
    getPickHandler()->reset();
    getPickHandler()->unload();
    return true;
}

void BaseViewer::fitNodeBBox(sofa::core::objectmodel::BaseNode * node )
{
    if(!currentCamera) return;
    if( node->f_bbox.getValue().isValid() && !node->f_bbox.getValue().isFlat() )
        currentCamera->fitBoundingBox(
            node->f_bbox.getValue().minBBox(),
            node->f_bbox.getValue().maxBBox()
        );

    redraw();
}

void BaseViewer::fitObjectBBox(sofa::core::objectmodel::BaseObject * object)
{
    if(!currentCamera) return;

    if( object->f_bbox.getValue().isValid() && !object->f_bbox.getValue().isFlat() )
        currentCamera->fitBoundingBox(object->f_bbox.getValue().minBBox(),
                object->f_bbox.getValue().maxBBox());
    else
    {
        if(object->getContext()->f_bbox.getValue().isValid() && !object->getContext()->f_bbox.getValue().isFlat()  )
        {
            currentCamera->fitBoundingBox(
                object->getContext()->f_bbox.getValue().minBBox(),
                object->getContext()->f_bbox.getValue().maxBBox());
        }
    }
    redraw();
}

} // namespace sofa::gui::common
