/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
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
    : m_simulationRoot(NULL)
    , m_currentCamera(NULL)
#ifndef SOFA_NO_OPENGL
    , texLogo(NULL)
#endif
    , m_doVideoRecording(false)
    , m_doDrawAxis(false)
    , m_backgroundColour(defaulttype::Vector3())
    , m_ambientColour(defaulttype::Vector3())
    , m_backgroundImageFilename("textures/SOFA_logo.bmp")
    , m_pickhandler(NULL)
    , m_screenshotDirectoryName(".")
{
    m_pickhandler = new PickHandler();
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
    return m_simulationRoot.get();
}
const std::string& BaseViewer::getSceneFileName()
{
    return m_sceneFileName;
}
void BaseViewer::setSceneFileName(const std::string &f)
{
    m_sceneFileName = f;
}

void BaseViewer::setScene(sofa::simulation::Node::SPtr scene, const char* filename /* = NULL */, bool /* = false */)
{
    std::string prefix = "";
    if (filename)
        prefix = sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(filename) + "_";
    setPrefix(prefix);

    m_sceneFileName = filename ? filename : std::string("default.scn");
    m_simulationRoot = scene;
    m_initTexturesDone = false;
}

void BaseViewer::setCameraMode(core::visual::VisualParams::CameraType mode)
{
    m_currentCamera->setCameraType(mode);
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
        m_pickhandler->setPickingMethod( gui::PickHandler::RAY_CASTING );
    else
        m_pickhandler->setPickingMethod( gui::PickHandler::SELECTION_BUFFER);
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

void BaseViewer::setPrefix(const std::string& prefix)
{
    const std::string fullPrefix = sofa::gui::BaseGUI::getScreenshotDirectoryPath() + "/" + prefix;
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
    if (!m_currentCamera)
        return;

    const defaulttype::Vec3d& camPosition = m_currentCamera->getPosition();
    const defaulttype::Quat& camOrientation = m_currentCamera->getOrientation();

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

    if (m_currentCamera)
        m_currentCamera->setView(position, orientation);

    redraw();
}

void BaseViewer::moveView(const defaulttype::Vector3& pos, const defaulttype::Quat &ori)
{
    if (!m_currentCamera)
        return;

    m_currentCamera->moveCamera(pos, ori);

    redraw();
}

void BaseViewer::newView()
{
    if (!m_currentCamera || !m_simulationRoot)
        return;

    m_currentCamera->setDefaultView(m_simulationRoot->getGravity());
}

void BaseViewer::resetView()
{
    redraw();
}

void BaseViewer::setBackgroundColour(float r, float g, float b)
{
    m_backgroundIndex = 2;
    m_backgroundColour[0] = r;
    m_backgroundColour[1] = g;
    m_backgroundColour[2] = b;
}

void BaseViewer::setBackgroundImage(std::string imageFileName)
{
    m_backgroundIndex = 0;

    if( sofa::helper::system::DataRepository.findFile(imageFileName) )
    {
        m_backgroundImageFilename = sofa::helper::system::DataRepository.getFile(imageFileName);
#ifndef SOFA_NO_OPENGL
        std::string extension = sofa::helper::system::SetDirectory::GetExtension(imageFileName.c_str());
        std::transform(extension.begin(),extension.end(),extension.begin(),::tolower );
        if(texLogo)
        {
            delete texLogo;
            texLogo = NULL;
        }
        helper::io::Image* image =  helper::io::Image::FactoryImage::getInstance()->createObject(extension,m_backgroundImageFilename);
        if( !image )
        {
            helper::vector<std::string> validExtensions;
            helper::io::Image::FactoryImage::getInstance()->uniqueKeys(std::back_inserter(validExtensions));
            std::cerr << "Could not create: " << imageFileName << std::endl;
            std::cerr << "Valid extensions: " << validExtensions << std::endl;
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
    return m_backgroundImageFilename;
}

PickHandler* BaseViewer::getPickHandler()
{
    return m_pickhandler;
}

bool BaseViewer::load()
{
    if (m_simulationRoot)
    {
        m_simulationRoot->get(m_currentCamera);
        if (!m_currentCamera)
        {
            m_currentCamera = sofa::core::objectmodel::New<component::visualmodel::InteractiveCamera>();
            m_currentCamera->setName(core::objectmodel::Base::shortName(m_currentCamera.get()));
            m_simulationRoot->addObject(m_currentCamera);
            m_currentCamera->p_position.forceSet();
            m_currentCamera->p_orientation.forceSet();
            m_currentCamera->bwdInit();
        }
        component::visualmodel::VisualStyle::SPtr visualStyle = NULL;
        m_simulationRoot->get(visualStyle);
        if (!visualStyle)
        {
            visualStyle = sofa::core::objectmodel::New<component::visualmodel::VisualStyle>();
            visualStyle->setName(core::objectmodel::Base::shortName(visualStyle.get()));

            core::visual::DisplayFlags* displayFlags = visualStyle->displayFlags.beginEdit();
            displayFlags->setShowVisualModels(sofa::core::visual::tristate::true_value);
            visualStyle->displayFlags.endEdit();

            m_simulationRoot->addObject(visualStyle);
            visualStyle->init();
        }

        m_currentCamera->setBoundingBox(m_simulationRoot->f_bbox.getValue().minBBox(),
                                        m_simulationRoot->f_bbox.getValue().maxBBox());

        // init pickHandler
        m_pickhandler->init(m_simulationRoot.get());

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
    if(!m_currentCamera) return;
    if( node->f_bbox.getValue().isValid() && !node->f_bbox.getValue().isFlat() )
        m_currentCamera->fitBoundingBox(
            node->f_bbox.getValue().minBBox(),
            node->f_bbox.getValue().maxBBox()
        );

    redraw();
}

void BaseViewer::fitObjectBBox(sofa::core::objectmodel::BaseObject * object)
{
    if(!m_currentCamera) return;

    if( object->f_bbox.getValue().isValid() && !object->f_bbox.getValue().isFlat() )
        m_currentCamera->fitBoundingBox(object->f_bbox.getValue().minBBox(),
                object->f_bbox.getValue().maxBBox());
    else
    {
        if(object->getContext()->f_bbox.getValue().isValid() && !object->getContext()->f_bbox.getValue().isFlat()  )
        {
            m_currentCamera->fitBoundingBox(
                object->getContext()->f_bbox.getValue().minBBox(),
                object->getContext()->f_bbox.getValue().maxBBox());
        }
    }
    redraw();
}

}
}

