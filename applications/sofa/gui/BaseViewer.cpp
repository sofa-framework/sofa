/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "BaseViewer.h"
#include "PickHandler.h"
#include "BaseGUI.h"

#include <sofa/helper/Factory.inl>
#include <SofaBaseVisual/VisualStyle.h>
#include <sofa/core/visual/DisplayFlags.h>

namespace sofa
{
namespace gui
{

BaseViewer::BaseViewer()
    : groot(NULL)
    , currentCamera(NULL)
#ifndef SOFA_NO_OPENGL
    , texLogo(NULL)
#endif
    , _video(false)
    , _axis(false)
    , backgroundColour(defaulttype::Vector3())
    , backgroundImageFile("textures/SOFA_logo.bmp")
    , ambientColour(defaulttype::Vector3())
    , pick(NULL)
    , _screenshotDirectory(".")
{
    pick = new PickHandler();
}

BaseViewer::~BaseViewer()
{
#ifndef SOFA_NO_OPENGL
   if(texLogo)
    {
        delete texLogo;
        texLogo = NULL;
    }
#endif
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

void BaseViewer::setScene(sofa::simulation::Node::SPtr scene, const char* filename /* = NULL */, bool /* = false */)
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

void BaseViewer::configure(sofa::component::configurationsetting::ViewerSetting* viewerConf)
{
    using namespace core::visual;
    if (viewerConf->cameraMode.getValue().getSelectedId() == VisualParams::ORTHOGRAPHIC_TYPE)
        setCameraMode(VisualParams::ORTHOGRAPHIC_TYPE);
    else
        setCameraMode(VisualParams::PERSPECTIVE_TYPE);
    if ( viewerConf->objectPickingMethod.getValue().getSelectedId() == gui::PickHandler::RAY_CASTING)
        pick->setPickingMethod( gui::PickHandler::RAY_CASTING );
    else
        pick->setPickingMethod( gui::PickHandler::SELECTION_BUFFER);
}

//Fonctions needed to take a screenshot
const std::string BaseViewer::screenshotName()
{
#ifndef SOFA_NO_OPENGL
    return capture.findFilename().c_str();
#else
    return "";
#endif
}

void BaseViewer::setPrefix(const std::string& prefix, bool prependDirectory)
{
    const std::string fullPrefix = (prependDirectory) ? sofa::gui::BaseGUI::getScreenshotDirectoryPath() + "/" + prefix
                                                      : prefix;
#ifndef SOFA_NO_OPENGL
    capture.setPrefix(fullPrefix);
#endif
#ifdef SOFA_HAVE_FFMPEG
    videoRecorder.setPrefix(fullPrefix);
#endif
}

void BaseViewer::screenshot(const std::string& filename, int compression_level)
{
#ifndef SOFA_NO_OPENGL
    capture.saveScreen(filename, compression_level);
#endif
}

void BaseViewer::getView(defaulttype::Vector3& pos, defaulttype::Quat& ori) const
{
    if (!currentCamera)
        return;

    const defaulttype::Vec3d& camPosition = currentCamera->getPosition();
    const defaulttype::Quat& camOrientation = currentCamera->getOrientation();

    pos[0] = camPosition[0];
    pos[1] = camPosition[1];
    pos[2] = camPosition[2];

    ori[0] = camOrientation[0];
    ori[1] = camOrientation[1];
    ori[2] = camOrientation[2];
    ori[3] = camOrientation[3];
}

void BaseViewer::setView(const defaulttype::Vector3& pos, const defaulttype::Quat &ori)
{
    defaulttype::Vec3d position;
    defaulttype::Quat orientation;
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

void BaseViewer::moveView(const defaulttype::Vector3& pos, const defaulttype::Quat &ori)
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
    _background = 2;
    backgroundColour[0] = r;
    backgroundColour[1] = g;
    backgroundColour[2] = b;
}

void BaseViewer::setBackgroundImage(std::string imageFileName)
{
    _background = 0;

    if( sofa::helper::system::DataRepository.findFile(imageFileName) )
    {
        backgroundImageFile = sofa::helper::system::DataRepository.getFile(imageFileName);
#ifndef SOFA_NO_OPENGL
        std::string extension = sofa::helper::system::SetDirectory::GetExtension(imageFileName.c_str());
        std::transform(extension.begin(),extension.end(),extension.begin(),::tolower );
        if(texLogo)
        {
            delete texLogo;
            texLogo = NULL;
        }
        helper::io::Image* image =  helper::io::Image::FactoryImage::getInstance()->createObject(extension,backgroundImageFile);
        if( !image )
        {
            helper::vector<std::string> validExtensions;
            helper::io::Image::FactoryImage::getInstance()->uniqueKeys(std::back_inserter(validExtensions));
            msg_error("BaseViewer") << "Could not create file '" << imageFileName <<"'" << msgendl
                                    << "Valid extensions: " << validExtensions ;
        }
        else
        {
            texLogo = new helper::gl::Texture( image );
            texLogo->init();
        }
#endif
    }
}

std::string BaseViewer::getBackgroundImage()
{
    return backgroundImageFile;
}

PickHandler* BaseViewer::getPickHandler()
{
    return pick;
}

bool BaseViewer::load()
{
    if (groot)
    {
        groot->get(currentCamera);
        if (!currentCamera)
        {
            currentCamera = sofa::core::objectmodel::New<component::visualmodel::InteractiveCamera>();
            currentCamera->setName(core::objectmodel::Base::shortName(currentCamera.get()));
            groot->addObject(currentCamera);
            //currentCamera->p_position.forceSet();
            //currentCamera->p_orientation.forceSet();
            currentCamera->bwdInit();
        }
        component::visualmodel::VisualStyle::SPtr visualStyle = NULL;
        groot->get(visualStyle);
        if (!visualStyle)
        {
            visualStyle = sofa::core::objectmodel::New<component::visualmodel::VisualStyle>();
            visualStyle->setName(core::objectmodel::Base::shortName(visualStyle.get()));

            core::visual::DisplayFlags* displayFlags = visualStyle->displayFlags.beginEdit();
            displayFlags->setShowVisualModels(sofa::core::visual::tristate::true_value);
            visualStyle->displayFlags.endEdit();

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





}
}

