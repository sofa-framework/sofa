#include "SofaViewer.h"
#include <sofa/helper/Factory.inl>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/core/visual/DisplayFlags.h>

namespace sofa
{
namespace gui
{
namespace qt
{
namespace viewer
{

SofaViewer::SofaViewer()
    : groot(NULL)
    , currentCamera(NULL)
    , m_isControlPressed(false)
    , _video(false)
    , _shadow(false)
    , _gl_shadow(true)
    , _axis(false)
    , backgroundColour(Vector3())
    , texLogo(NULL)
    , ambientColour(Vector3())
{
    colourPickingRenderCallBack = ColourPickingRenderCallBack(this);
}

SofaViewer::~SofaViewer()
{
    if(texLogo)
    {
        delete texLogo;
        texLogo = NULL;
    }
}

sofa::simulation::Node* SofaViewer::getScene()
{
    return groot;
}
const std::string& SofaViewer::getSceneFileName()
{
    return sceneFileName;
}
void SofaViewer::setSceneFileName(const std::string &f)
{
    sceneFileName = f;
}

void SofaViewer::setScene(sofa::simulation::Node* scene, const char* filename /* = NULL */, bool /* = false */)
{
    std::string file =
        filename ? sofa::helper::system::SetDirectory::GetFileName(
                filename) : std::string();
    std::string screenshotPrefix =
        sofa::helper::system::SetDirectory::GetParentDir(
                sofa::helper::system::DataRepository.getFirstPath().c_str())
        + std::string("/share/screenshots/") + file
        + std::string("_");
    capture.setPrefix(screenshotPrefix);
#ifdef SOFA_HAVE_FFMPEG
    videoRecorder.setPrefix(screenshotPrefix);
#endif //SOFA_HAVE_FFMPEG
    sceneFileName = filename ? filename : std::string("default.scn");
    groot = scene;
    initTexturesDone = false;
    _stereoEnabled = false;
    _stereoShift = 1.0;
    _binocularModeEnabled = false;

    //Camera initialization
    if (groot)
    {
        groot->get(currentCamera);
        if (!currentCamera)
        {
            currentCamera = new component::visualmodel::InteractiveCamera();
            currentCamera->setName(core::objectmodel::Base::shortName(currentCamera));
            groot->addObject(currentCamera);
            currentCamera->p_position.forceSet();
            currentCamera->p_orientation.forceSet();
            currentCamera->bwdInit();

        }
        component::visualmodel::VisualStyle* visualStyle = NULL;
        groot->get(visualStyle);
        if (!visualStyle)
        {
            visualStyle = new component::visualmodel::VisualStyle();
            visualStyle->setName(core::objectmodel::Base::shortName(visualStyle));

            core::visual::DisplayFlags* displayFlags = visualStyle->displayFlags.beginEdit();
            displayFlags->setShowVisualModels(sofa::core::visual::tristate::true_value);
            visualStyle->displayFlags.endEdit();

            groot->addObject(visualStyle);
            visualStyle->init();
        }

        currentCamera->setBoundingBox(groot->f_bbox.getValue().minBBox(), groot->f_bbox.getValue().maxBBox());

        // init pickHandler
        pick.init();
        pick.setColourRenderCallback(&colourPickingRenderCallBack);

    }



}

void SofaViewer::setCameraMode(core::visual::VisualParams::CameraType mode)
{
    currentCamera->setCameraType(mode);
}

bool SofaViewer::ready()
{
    return true;
}

void SofaViewer::configure(sofa::component::configurationsetting::ViewerSetting* viewerConf)
{
    using namespace core::visual;
    if (viewerConf->cameraMode.getValue().getSelectedId() == VisualParams::ORTHOGRAPHIC_TYPE)
        setCameraMode(VisualParams::ORTHOGRAPHIC_TYPE);
    else
        setCameraMode(VisualParams::PERSPECTIVE_TYPE);
    if ( viewerConf->objectPickingMethod.getValue().getSelectedId() == gui::PickHandler::RAY_CASTING)
        pick.setPickingMethod( gui::PickHandler::RAY_CASTING );
    else
        pick.setPickingMethod( gui::PickHandler::SELECTION_BUFFER);
}
//Fonctions needed to take a screenshot
const std::string SofaViewer::screenshotName()
{
    return capture.findFilename().c_str();
}

void SofaViewer::setPrefix(const std::string filename)
{
    capture.setPrefix(filename);
}

void SofaViewer::screenshot(const std::string filename, int compression_level)
{
    capture.saveScreen(filename, compression_level);
}

void SofaViewer::getView(Vec3d& pos, Quat& ori) const
{
    if (!currentCamera)
        return;

    const Vec3d& camPosition = currentCamera->getPosition();
    const Quat& camOrientation = currentCamera->getOrientation();

    pos[0] = camPosition[0];
    pos[1] = camPosition[1];
    pos[2] = camPosition[2];

    ori[0] = camOrientation[0];
    ori[1] = camOrientation[1];
    ori[2] = camOrientation[2];
    ori[3] = camOrientation[3];
}

void SofaViewer::setView(const Vec3d& pos, const Quat &ori)
{
    Vec3d position;
    Quat orientation;
    for (unsigned int i=0 ; i<3 ; i++)
    {
        position[i] = pos[i];
        orientation[i] = ori[i];
    }
    orientation[3] = ori[3];

    if (currentCamera)
        currentCamera->setView(position, orientation);

    getQWidget()->update();
}

void SofaViewer::moveView(const Vec3d& pos, const Quat &ori)
{
    if (!currentCamera)
        return;

    currentCamera->moveCamera(pos, ori);
    getQWidget()->update();
}

void SofaViewer::newView()
{
    if (!currentCamera || !groot)
        return;

    currentCamera->setDefaultView(groot->getGravity());
}

void SofaViewer::resetView()
{
    getQWidget()->update();
}

void SofaViewer::keyPressEvent(QKeyEvent * e)
{
    sofa::core::objectmodel::KeypressedEvent kpe(e->key());
    currentCamera->manageEvent(&kpe);

    switch (e->key())
    {
    case Qt::Key_T:
    {
        if (currentCamera->getCameraType() == core::visual::VisualParams::ORTHOGRAPHIC_TYPE)
            setCameraMode(core::visual::VisualParams::PERSPECTIVE_TYPE);
        else
            setCameraMode(core::visual::VisualParams::ORTHOGRAPHIC_TYPE);
        break;
    }
    case Qt::Key_Shift:
        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT,viewport);
        pick.activateRay(viewport[2],viewport[3]);
        break;
    case Qt::Key_B:
        // --- change background
    {
        _background = (_background + 1) % 3;
        break;
    }
    case Qt::Key_L:
        // --- draw shadows
    {
        if (_gl_shadow)
            _shadow = !_shadow;
        break;
    }
    case Qt::Key_R:
        // --- draw axis
    {
        _axis = !_axis;
        break;
    }
    case Qt::Key_S:
    {
        screenshot(capture.findFilename());
        break;
    }
    case Qt::Key_V:
        // --- save video
    {
        if(!_video)
        {
            switch (SofaVideoRecorderManager::getInstance()->getRecordingType())
            {
            case SofaVideoRecorderManager::SCREENSHOTS :
                break;
            case SofaVideoRecorderManager::MOVIE :
            {
#ifdef SOFA_HAVE_FFMPEG
                SofaVideoRecorderManager* videoManager = SofaVideoRecorderManager::getInstance();
                unsigned int framerate = videoManager->getFramerate(); //img.s-1
                unsigned int freq = ceil(1000/framerate); //ms
                unsigned int bitrate = videoManager->getBitrate();
                std::string videoFilename = videoRecorder.findFilename(videoManager->getCodecExtension());
                videoRecorder.init( videoFilename, framerate, bitrate);
                captureTimer.start(freq);
#endif

                break;
            }
            default :
                break;
            }

        }
        else
        {
            switch (SofaVideoRecorderManager::getInstance()->getRecordingType())
            {
            case SofaVideoRecorderManager::SCREENSHOTS :
                break;
            case SofaVideoRecorderManager::MOVIE :
            {
                captureTimer.stop();
#ifdef SOFA_HAVE_FFMPEG
                videoRecorder.finishVideo();
#endif //SOFA_HAVE_FFMPEG
                break;
            }
            default :
                break;
            }
        }

        _video = !_video;
        //capture.setCounter();

        break;
    }
    case Qt::Key_W:
        // --- save current view
    {
        saveView();
        break;
    }
    case Qt::Key_F1:
        // --- enable stereo mode
    {
        _stereoEnabled = !_stereoEnabled;
        std::cout << "Stereoscopic View Enabled" << std::endl;
        break;
    }
    case Qt::Key_F2:
        // --- reduce shift distance
    {
        _stereoShift -= 0.1;
        break;
    }
    case Qt::Key_F3:
        // --- increase shift distance
    {
        _stereoShift += 0.1;
        break;
    }
    case Qt::Key_F5:
        // --- enable binocular mode
    {
        std::cout << "Binocular View Enabled" << std::endl;
        _binocularModeEnabled = !_binocularModeEnabled;
        break;
    }
    case Qt::Key_Control:
    {
        m_isControlPressed = true;
        //cerr<<"QtViewer::keyPressEvent, CONTROL pressed"<<endl;
        break;
    }
    default:
    {
        e->ignore();
    }
    }
}

void SofaViewer::keyReleaseEvent(QKeyEvent * e)
{
    sofa::core::objectmodel::KeyreleasedEvent kre(e->key());
    currentCamera->manageEvent(&kre);

    switch (e->key())
    {
    case Qt::Key_Shift:
        pick.deactivateRay();

        break;
    case Qt::Key_Control:
    {
        m_isControlPressed = false;

        // Send Control Release Info to a potential ArticulatedRigid Instrument
        sofa::core::objectmodel::MouseEvent mouseEvent(
            sofa::core::objectmodel::MouseEvent::Reset);
        if (groot)
            groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
    }
    default:
    {
        e->ignore();
    }
    }

    if (isControlPressed())
    {
        sofa::core::objectmodel::KeyreleasedEvent keyEvent(e->key());
        if (groot)
            groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
    }

}

bool SofaViewer::isControlPressed() const
{
    return m_isControlPressed;
}

// ---------------------- Here are the Mouse controls   ----------------------
void SofaViewer::wheelEvent(QWheelEvent *e)
{
    //<CAMERA API>
    sofa::core::objectmodel::MouseEvent me(sofa::core::objectmodel::MouseEvent::Wheel,e->delta());
    currentCamera->manageEvent(&me);

    getQWidget()->update();
#ifndef SOFA_GUI_INTERACTION
    if (groot)
        groot->propagateEvent(core::ExecParams::defaultInstance(), &me);
#endif
}

void SofaViewer::mouseMoveEvent ( QMouseEvent *e )
{
    //<CAMERA API>
    sofa::core::objectmodel::MouseEvent me(sofa::core::objectmodel::MouseEvent::Move,e->x(), e->y());
    currentCamera->manageEvent(&me);

    getQWidget()->update();
#ifndef SOFA_GUI_INTERACTION
    if (groot)
        groot->propagateEvent(core::ExecParams::defaultInstance(), &me);
#endif
}

void SofaViewer::mousePressEvent ( QMouseEvent * e)
{
    //<CAMERA API>
    sofa::core::objectmodel::MouseEvent* mEvent = NULL;
    if (e->button() == Qt::LeftButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftPressed, e->x(), e->y());
    else if (e->button() == Qt::RightButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightPressed, e->x(), e->y());
    else if (e->button() == Qt::MidButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddlePressed, e->x(), e->y());
    currentCamera->manageEvent(mEvent);

    getQWidget()->update();
#ifndef SOFA_GUI_INTERACTION
    if (groot)
        groot->propagateEvent(core::ExecParams::defaultInstance(), mEvent);
#endif
}

void SofaViewer::mouseReleaseEvent ( QMouseEvent * e)
{
    //<CAMERA API>
    sofa::core::objectmodel::MouseEvent* mEvent = NULL;
    if (e->button() == Qt::LeftButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftReleased, e->x(), e->y());
    else if (e->button() == Qt::RightButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightReleased, e->x(), e->y());
    else if (e->button() == Qt::MidButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddleReleased, e->x(), e->y());
    currentCamera->manageEvent(mEvent);

    getQWidget()->update();
#ifndef SOFA_GUI_INTERACTION
    if (groot)
        groot->propagateEvent(core::ExecParams::defaultInstance(), mEvent);
#endif
}

void SofaViewer::mouseEvent(QMouseEvent *e)
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);

    MousePosition mousepos;
    mousepos.screenWidth  = viewport[2];
    mousepos.screenHeight = viewport[3];
    mousepos.x      = e->x();
    mousepos.y      = e->y();

    if (e->state() & Qt::ShiftButton)
    {

        pick.activateRay(viewport[2],viewport[3]);
        pick.updateMouse2D( mousepos );

        //_sceneTransform.ApplyInverse();
        switch (e->type())
        {
        case QEvent::MouseButtonPress:

            if (e->button() == Qt::LeftButton)
            {
                pick.handleMouseEvent(PRESSED, LEFT);
            }
            else if (e->button() == Qt::RightButton) // Shift+Rightclick to remove triangles
            {
                pick.handleMouseEvent(PRESSED, RIGHT);
            }
            else if (e->button() == Qt::MidButton) // Shift+Midclick (by 2 steps defining 2 input points) to cut from one point to another
            {
                pick.handleMouseEvent(PRESSED, MIDDLE);
            }
            break;
        case QEvent::MouseButtonRelease:
            //if (e->button() == Qt::LeftButton)
        {

            if (e->button() == Qt::LeftButton)
            {
                pick.handleMouseEvent(RELEASED, LEFT);
            }
            else if (e->button() == Qt::RightButton)
            {
                pick.handleMouseEvent(RELEASED, RIGHT);
            }
            else if (e->button() == Qt::MidButton)
            {
                pick.handleMouseEvent(RELEASED, MIDDLE);
            }
        }
        break;
        default:
            break;
        }
        moveRayPickInteractor(e->x(), e->y());
    }
    else
    {
        pick.activateRay(viewport[2],viewport[3]);
    }

}

void SofaViewer::captureEvent()
{
    if (_video)
    {
        switch (SofaVideoRecorderManager::getInstance()->getRecordingType())
        {
        case SofaVideoRecorderManager::SCREENSHOTS :
            if(!captureTimer.isActive())
                screenshot(capture.findFilename(), 1);
            break;
        case SofaVideoRecorderManager::MOVIE :
            if(captureTimer.isActive())
            {
#ifdef SOFA_HAVE_FFMPEG
                videoRecorder.addFrame();
#endif //SOFA_HAVE_FFMPEG
            }
            break;
        default :
            break;
        }
    }
}

void SofaViewer::setBackgroundColour(float r, float g, float b)
{
    _background = 2;
    backgroundColour[0] = r;
    backgroundColour[1] = g;
    backgroundColour[2] = b;
}

void SofaViewer::setBackgroundImage(std::string imageFileName)
{
    _background = 0;

    if( sofa::helper::system::DataRepository.findFile(imageFileName) )
    {
        backgroundImageFile = sofa::helper::system::DataRepository.getFile(imageFileName);
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
            std::cerr << "Could not create: " << imageFileName << std::endl;
            std::cerr << "Valid extensions: " << validExtensions << std::endl;
        }
        else
        {
            texLogo = new helper::gl::Texture( image );
            texLogo->init();

        }

    }
}

std::string SofaViewer::getBackgroundImage()
{
    return backgroundImageFile;
}
PickHandler* SofaViewer::getPickHandler()
{
    return &pick;
}

void SofaViewer::fitNodeBBox(sofa::core::objectmodel::BaseNode * node )
{
    if(!currentCamera) return;
    if( node->f_bbox.getValue().isValid() && !node->f_bbox.getValue().isFlat() )
        currentCamera->fitBoundingBox(
            node->f_bbox.getValue().minBBox(),
            node->f_bbox.getValue().maxBBox()
        );

    this->getQWidget()->update();

}

void SofaViewer::fitObjectBBox(sofa::core::objectmodel::BaseObject * object)
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
    this->getQWidget()->update();

}





}
}
}
}
