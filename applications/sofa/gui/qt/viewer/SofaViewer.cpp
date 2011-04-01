#include "SofaViewer.h"
#include <sofa/helper/Factory.inl>

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
    sceneBBoxIsValid = false;

    //Camera initialization
    if (groot)
    {
        groot->get(currentCamera);
        if (!currentCamera)
        {
            currentCamera = new component::visualmodel::InteractiveCamera();
            groot->addObject(currentCamera);
            currentCamera->p_position.forceSet();
            currentCamera->p_orientation.forceSet();
            currentCamera->init();

        }
        sofa::defaulttype::Vector3 minBBox, maxBBox;

        sofa::simulation::getSimulation()->computeBBox(simulation::getSimulation()->getVisualRoot(), minBBox.ptr(),maxBBox.ptr());

        currentCamera->setBoundingBox(minBBox, maxBBox);

    }

    // init pickHandler
    pick.init();
    pick.setColourRenderCallback(&colourPickingRenderCallBack);

}

void SofaViewer::setCameraMode(component::visualmodel::BaseCamera::CameraType mode)
{
    currentCamera->setCameraType(mode);
}

bool SofaViewer::ready()
{
    return true;
}

void SofaViewer::configure(sofa::component::configurationsetting::ViewerSetting* viewerConf)
{
    if (viewerConf->cameraMode.getValue().getSelectedId() == component::visualmodel::BaseCamera::ORTHOGRAPHIC_TYPE)
        setCameraMode(component::visualmodel::BaseCamera::ORTHOGRAPHIC_TYPE);
    else
        setCameraMode(component::visualmodel::BaseCamera::PERSPECTIVE_TYPE);
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
        if (currentCamera->getCameraType() == component::visualmodel::BaseCamera::ORTHOGRAPHIC_TYPE)
            setCameraMode(component::visualmodel::BaseCamera::PERSPECTIVE_TYPE);
        else
            setCameraMode(component::visualmodel::BaseCamera::ORTHOGRAPHIC_TYPE);
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

    if (groot)
        groot->propagateEvent(core::ExecParams::defaultInstance(), &me);
}

void SofaViewer::mouseMoveEvent ( QMouseEvent *e )
{
    //<CAMERA API>
    sofa::core::objectmodel::MouseEvent me(sofa::core::objectmodel::MouseEvent::Move,e->x(), e->y());
    currentCamera->manageEvent(&me);

    getQWidget()->update();

    if (groot)
        groot->propagateEvent(core::ExecParams::defaultInstance(), &me);
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
    if (groot)
        groot->propagateEvent(core::ExecParams::defaultInstance(), mEvent);
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
    if (groot)
        groot->propagateEvent(core::ExecParams::defaultInstance(), mEvent);
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

// ---------------------- Here are the controls for instruments  ----------------------

void SofaViewer::moveLaparoscopic( QMouseEvent *e)
{
    int index_instrument = simulation::getSimulation()->instrumentInUse.getValue();
    if (index_instrument < 0 || index_instrument > (int)simulation::getSimulation()->instruments.size()) return;

    simulation::Node *instrument = simulation::getSimulation()->instruments[index_instrument];
    if (instrument == NULL) return;

    int eventX = e->x();
    int eventY = e->y();

    std::vector< sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* > instruments;
    instrument->getTreeObjects<sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>, std::vector< sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* > >(&instruments);

    if (!instruments.empty())
    {
        sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* instrument = instruments[0];
        switch (e->type())
        {
        case QEvent::MouseButtonPress:
            // Mouse left button is pushed
            if (e->button() == Qt::LeftButton)
            {
                _navigationMode = BTLEFT_MODE;
                _mouseInteractorMoving = true;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
            // Mouse right button is pushed
            else if (e->button() == Qt::RightButton)
            {
                _navigationMode = BTRIGHT_MODE;
                _mouseInteractorMoving = true;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
            // Mouse middle button is pushed
            else if (e->button() == Qt::MidButton)
            {
                _navigationMode = BTMIDDLE_MODE;
                _mouseInteractorMoving = true;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
            break;

        case QEvent::MouseMove:
            //
            break;

        case QEvent::MouseButtonRelease:
            // Mouse left button is released
            if (e->button() == Qt::LeftButton)
            {
                if (_mouseInteractorMoving)
                {
                    _mouseInteractorMoving = false;
                }
            }
            // Mouse right button is released
            else if (e->button() == Qt::RightButton)
            {
                if (_mouseInteractorMoving)
                {
                    _mouseInteractorMoving = false;
                }
            }
            // Mouse middle button is released
            else if (e->button() == Qt::MidButton)
            {
                if (_mouseInteractorMoving)
                {
                    _mouseInteractorMoving = false;
                    //static_cast<sofa::simulation::Node*>(instrument->getContext())->execute<sofa::simulation::GrabVisitor>();
                }
            }
            break;

        default:
            break;
        }
        if (_mouseInteractorMoving)
        {
            {
                helper::WriteAccessor<Data<sofa::defaulttype::LaparoscopicRigidTypes::VecCoord> > instrumentX = *instrument->write(core::VecCoordId::position());
                if (_navigationMode == BTLEFT_MODE)
                {
                    int dx = eventX - _mouseInteractorSavedPosX;
                    int dy = eventY - _mouseInteractorSavedPosY;
                    if (dx || dy)
                    {
                        instrumentX[0].getOrientation() = instrumentX[0].getOrientation() * Quat(Vector3(0,1,0),dx*0.001) * Quat(Vector3(0,0,1),dy*0.001);
                        _mouseInteractorSavedPosX = eventX;
                        _mouseInteractorSavedPosY = eventY;
                    }
                }
                else if (_navigationMode == BTMIDDLE_MODE)
                {
                    int dx = eventX - _mouseInteractorSavedPosX;
                    int dy = eventY - _mouseInteractorSavedPosY;
                    if (dx || dy)
                    {
                        _mouseInteractorSavedPosX = eventX;
                        _mouseInteractorSavedPosY = eventY;
                    }
                }
                else if (_navigationMode == BTRIGHT_MODE)
                {
                    int dx = eventX - _mouseInteractorSavedPosX;
                    int dy = eventY - _mouseInteractorSavedPosY;
                    if (dx || dy)
                    {
                        instrumentX[0].getTranslation() += (dy)*0.01;
                        instrumentX[0].getOrientation() = instrumentX[0].getOrientation() * Quat(Vector3(1,0,0),dx*0.001);
                        _mouseInteractorSavedPosX = eventX;
                        _mouseInteractorSavedPosY = eventY;
                    }
                }
            }
            sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor(core::MechanicalParams::defaultInstance()).execute(instrument->getContext());
            sofa::simulation::UpdateMappingVisitor(core::ExecParams::defaultInstance()).execute(instrument->getContext());
            //static_cast<sofa::simulation::Node*>(instrument->getContext())->execute<sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor>();
            //static_cast<sofa::simulation::Node*>(instrument->getContext())->execute<sofa::simulation::UpdateMappingVisitor>();
        }
        getQWidget()->update();
    }
    else
    {
        std::vector< component::controller::Controller* > bc;
        instrument->getTreeObjects<component::controller::Controller, std::vector< component::controller::Controller* > >(&bc);

        if (!bc.empty())
        {
            switch (e->type())
            {
            case QEvent::MouseButtonPress:
                // Mouse left button is pushed
                if (e->button() == Qt::LeftButton)
                {
                    sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::LeftPressed, eventX, eventY);
                    if (groot)
                        groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
                }
                // Mouse right button is pushed
                else if (e->button() == Qt::RightButton)
                {
                    sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::RightPressed, eventX, eventY);
                    if (groot)
                        groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
                }
                // Mouse middle button is pushed
                else if (e->button() == Qt::MidButton)
                {
                    sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::MiddlePressed, eventX, eventY);
                    if (groot)
                        groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
                }
                break;

            case QEvent::MouseMove:
            {
                if (e->state()&(Qt::LeftButton|Qt::RightButton|Qt::MidButton))
                {
                    sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::Move, eventX, eventY);
                    if (groot)
                        groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
                }
            }
            break;

            case QEvent::MouseButtonRelease:
                // Mouse left button is released
                if (e->button() == Qt::LeftButton)
                {
                    sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::LeftReleased, eventX, eventY);
                    if (groot)
                        groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
                }
                // Mouse right button is released
                else if (e->button() == Qt::RightButton)
                {
                    sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::RightReleased, eventX, eventY);
                    if (groot)
                        groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
                }
                // Mouse middle button is released
                else if (e->button() == Qt::MidButton)
                {
                    sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::MiddleReleased, eventX, eventY);
                    if (groot)
                        groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
                }
                break;

            default:
                break;
            }

            getQWidget()->update();
        }
    }
}

void SofaViewer::moveLaparoscopic(QWheelEvent *e)
{
    int index_instrument =
        simulation::getSimulation()->instrumentInUse.getValue();
    if (index_instrument < 0 || index_instrument
        > (int) simulation::getSimulation()->instruments.size())
        return;

    simulation::Node *instrument =
        simulation::getSimulation()->instruments[index_instrument];
    if (instrument == NULL)
        return;

    std::vector<component::controller::Controller*> bc;
    instrument->getTreeObjects<component::controller::Controller,
               std::vector<component::controller::Controller*> > (&bc);

    if (!bc.empty())
    {
        sofa::core::objectmodel::MouseEvent mouseEvent(
            sofa::core::objectmodel::MouseEvent::Wheel, e->delta());
        if (groot)
            groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
        getQWidget()->update();
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





}
}
}
}
