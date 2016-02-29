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
#include "SofaViewer.h"
#include <sofa/helper/Factory.inl>
#include <SofaBaseVisual/VisualStyle.h>
#include <sofa/core/visual/DisplayFlags.h>

using sofa::core::objectmodel::KeypressedEvent;
using sofa::core::objectmodel::KeyreleasedEvent;
using sofa::core::objectmodel::MouseEvent;
using sofa::gui::BaseViewer ;
using sofa::core::visual::VisualParams ;
using sofa::component::visualmodel::BaseCamera ;
using sofa::core::ExecParams ;

namespace sofa
{
namespace gui
{
namespace qt
{
namespace viewer
{

SofaViewer::SofaViewer() : BaseViewer()
{
    colourPickingRenderCallBack = ColourPickingRenderCallBack(this);
    m_state = SofaViewer::STATE_CAMERAMANIPULATION ;
}

SofaViewer::~SofaViewer()
{
}

void SofaViewer::redraw()
{
    getQWidget()->update();
}

void SofaViewer::toggleVideoRecording(){
    if(!m_doVideoRecording)
    {
        switch (SofaVideoRecorderManager::getInstance()->getRecordingType())
        {
        case SofaVideoRecorderManager::SCREENSHOTS :
            break;
        case SofaVideoRecorderManager::MOVIE :
        {
           #ifdef SOFA_HAVE_FFMPEG
                SofaVideoRecorderManager* videoManager = SofaVideoRecorderManager::getInstance();
                unsigned int bitrate = videoManager->getBitrate();
                unsigned int framerate = videoManager->getFramerate();
                std::string videoFilename = videoRecorder.findFilename(videoManager->getCodecExtension());
                videoRecorder.init( videoFilename, framerate, bitrate, videoManager->getCodecName());
           #else
                std::cout << "This version of SOFA has not been compiled with "
                             "video recording support. Try to enable FFMPEG support" << std::endl ;
           #endif // SOFA_HAVE_FFMPEG

            break;
        }
        default :
            break;
        }
        if (SofaVideoRecorderManager::getInstance()->realtime())
        {
            unsigned int framerate = SofaVideoRecorderManager::getInstance()->getFramerate();
            std::cout << "Starting capture timer ( " << framerate << " Hz )" << std::endl;
            unsigned int interv = (1000+framerate-1)/framerate;
            captureTimer.start(interv);
        }

    }
    else
    {
        if(captureTimer.isActive())
        {
            std::cout << "Stopping capture timer" << std::endl;
            captureTimer.stop();
        }
        switch (SofaVideoRecorderManager::getInstance()->getRecordingType())
        {
        case SofaVideoRecorderManager::SCREENSHOTS :
            break;
        case SofaVideoRecorderManager::MOVIE :
        {
            #ifdef SOFA_HAVE_FFMPEG
                videoRecorder.finishVideo();
            #endif
            break;

        }
        default :
            break;
        }
    }

    m_doVideoRecording = !m_doVideoRecording;
}

void  SofaViewer::toggleCameraMode(){
    if (currentCamera->getCameraType() == VisualParams::ORTHOGRAPHIC_TYPE){
        setCameraMode(VisualParams::PERSPECTIVE_TYPE);
    }else{
        setCameraMode(VisualParams::ORTHOGRAPHIC_TYPE);
    }
}

bool SofaViewer::keyPressEvent_p(QKeyEvent * e)
{
    KeypressedEvent kpe(e->key());

    if(m_state==STATE_SCENEFORWARDING){
        if(e->key() == Qt::Key_Control )
            return true;

        if (m_simulationRoot)
            m_simulationRoot->propagateEvent(ExecParams::defaultInstance(), &kpe);
    }else if(m_state==STATE_PICKING){
        // Nothing to do

    }else if(m_state==STATE_CAMERAMANIPULATION){
        if(currentCamera)
            currentCamera->manageEvent(&kpe);

        switch (e->key())
        {
        case Qt::Key_A: {
            switchAxisViewing();
            break ;
        }
        case Qt::Key_C: {
            viewAll() ;
            break ;
        }
        case Qt::Key_B: {
            // TODO(dmarchal): do we really need a short cut for that ?
            // it should probably be much more suited as a configuration option
            // theme or something.
            m_backgroundIndex = (m_backgroundIndex + 1) % 3;
            break;
        }
        case Qt::Key_R: {
            toogleBoundingBoxDraw();
            break;
        }
        case Qt::Key_S: {
            screenshot(capture.findFilename());
            break;
        }
        case Qt::Key_T: {
            toggleCameraMode();
            break;
        }
        case Qt::Key_V: {
            toggleVideoRecording() ;
            break;
        }
        case Qt::Key_W: {
            saveView();
            break;
        }
        case Qt::Key_F1:
            // --- enable stereo mode
        {
            currentCamera->setStereoEnabled(!currentCamera->getStereoEnabled());
            std::cout << "Stereoscopic View " << (currentCamera->getStereoEnabled() ? "Enabled" : "Disabled") << std::endl;
            break;
        }
        case Qt::Key_F2:
            // --- reduce shift distance
        {
            currentCamera->setStereoShift(currentCamera->getStereoShift()-0.1);
            std::cout << "Stereo separation = " << currentCamera->getStereoShift() << std::endl;
            break;
        }
        case Qt::Key_F3:
            // --- increase shift distance
        {
            currentCamera->setStereoShift(currentCamera->getStereoShift()+0.1);
            std::cout << "Stereo separation = " << currentCamera->getStereoShift() << std::endl;
            break;
        }
        case Qt::Key_F4:
        {
            // --- Switch between parallax and toedIn stereovision
            switch (currentCamera->getStereoStrategy()) {
            case BaseCamera::PARALLEL:
                currentCamera->setStereoStrategy(BaseCamera::TOEDIN);
                std::cout << "Stereo Strategy: TOEDIN" << std::endl;
                break;
            case BaseCamera::TOEDIN:
                currentCamera->setStereoStrategy(BaseCamera::PARALLEL);
                std::cout << "Stereo Strategy: Parallel" << std::endl;
                break;
            default:
                currentCamera->setStereoStrategy(BaseCamera::PARALLEL);
                break;
            }
            break;
        }
        case Qt::Key_F5:
            // --- enable binocular mode
        {
            //_stereoMode = (StereoMode)(((int)_stereoMode+1)%(int)NB_STEREO_MODES);
            currentCamera->setStereoMode((sofa::component::visualmodel::BaseCamera::StereoMode)(((int)currentCamera->getStereoMode()+1)%(int)sofa::component::visualmodel::BaseCamera::NB_STEREO_MODES));
            switch (currentCamera->getStereoMode())
            {
            case BaseCamera::STEREO_INTERLACED:
                std::cout << "Stereo mode: Interlaced" << std::endl;
                break;
            case BaseCamera::STEREO_SIDE_BY_SIDE:
                std::cout << "Stereo mode: Side by Side" << std::endl; break;
            case BaseCamera::STEREO_SIDE_BY_SIDE_HALF:
                std::cout << "Stereo mode: Side by Side Half" << std::endl; break;
            case BaseCamera::STEREO_FRAME_PACKING:
                std::cout << "Stereo mode: Frame Packing" << std::endl; break;
            case BaseCamera::STEREO_TOP_BOTTOM:
                std::cout << "Stereo mode: Top Bottom" << std::endl; break;
            case BaseCamera::STEREO_TOP_BOTTOM_HALF:
                std::cout << "Stereo mode: Top Bottom Half" << std::endl; break;
            case BaseCamera::STEREO_AUTO:
                std::cout << "Stereo mode: Automatic" << std::endl; break;
            case BaseCamera::STEREO_NONE:
                std::cout << "Stereo mode: None" << std::endl; break;
            default:
                std::cout << "Stereo mode: INVALID" << std::endl; break;
                break;
            }
            break;
        }
        case Qt::Key_Control:
            m_state = STATE_SCENEFORWARDING ;
            break;

        case Qt::Key_Shift:
            m_state = STATE_PICKING ;
            GLint viewport[4];
            glGetIntegerv(GL_VIEWPORT,viewport);
            getPickHandler()->activateRay(viewport[2],viewport[3], m_simulationRoot.get());
            break;

        default:
            break;
        }
    }// END STATE_MANIPULATION

    return true;
}

bool SofaViewer::keyReleaseEvent_p(QKeyEvent * e)
{
    KeyreleasedEvent kre(e->key());

    if(m_state==STATE_SCENEFORWARDING){
        if(e->key() == Qt::Key_Control){
            m_state = STATE_CAMERAMANIPULATION;
            return true;
        }
        if (m_simulationRoot)
            m_simulationRoot->propagateEvent(ExecParams::defaultInstance(), &kre);

        // TODO(dmarchal): this seems a kind of patchy... should be remove.
        // A Control release that rise a mouseEvent sound real weird to me.
        // Send Control Release Info to a potential ArticulatedRigid Instrument
        MouseEvent mouseEvent(MouseEvent::Reset);
        if (m_simulationRoot)
            m_simulationRoot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
    }else if(m_state==STATE_PICKING){
        if(e->key() == Qt::Key_Shift){
            getPickHandler()->deactivateRay();
            m_state = STATE_CAMERAMANIPULATION;
        }
    }else if(m_state==STATE_CAMERAMANIPULATION){

    }

    return true;
}

// ---------------------- Here are the Mouse controls   ----------------------
bool SofaViewer::wheelEvent_p(QWheelEvent *e)
{
    MouseEvent me(MouseEvent::Wheel,e->delta());

    switch(m_state){
    case STATE_SCENEFORWARDING:
        if (m_simulationRoot)
            m_simulationRoot->propagateEvent(ExecParams::defaultInstance(), &me);
        break;
    case STATE_PICKING:
        break;
    case STATE_CAMERAMANIPULATION:
        if (currentCamera)
            currentCamera->manageEvent(&me);
        break;
    default:
        break;
    }

    return true;
}

bool SofaViewer::mouseMoveEvent_p( QMouseEvent *e )
{
    MouseEvent me(MouseEvent::Move,e->x(), e->y());

    switch(m_state){
    case STATE_SCENEFORWARDING:
        if (m_simulationRoot)
            m_simulationRoot->propagateEvent(ExecParams::defaultInstance(), &me);
        break;
    case STATE_PICKING:
        updatePicking(e);
        break;
    case STATE_CAMERAMANIPULATION:
        if (currentCamera)
            currentCamera->manageEvent(&me);
        break;
    default:
        break;
    }

    return true;
}

bool SofaViewer::mousePressEvent_p( QMouseEvent * e)
{
    MouseEvent* mEvent = NULL;
    if (e->button() == Qt::LeftButton)
        mEvent = new MouseEvent(MouseEvent::LeftPressed,
                                e->x(), e->y());
    else if (e->button() == Qt::RightButton)
        mEvent = new MouseEvent(MouseEvent::RightPressed,
                                e->x(), e->y());
    else if (e->button() == Qt::MidButton)
        mEvent = new MouseEvent(MouseEvent::MiddlePressed,
                                e->x(), e->y());
    else{
        // A fallback event to rules them all...
        mEvent = new MouseEvent(MouseEvent::AnyExtraButtonPressed,
                                e->x(), e->y());
    }

    switch(m_state){
    case STATE_SCENEFORWARDING:
        if (m_simulationRoot)
            m_simulationRoot->propagateEvent(ExecParams::defaultInstance(), mEvent);
        break;
    case STATE_PICKING:
        updatePicking(e);
        break;
    case STATE_CAMERAMANIPULATION:
        if (currentCamera)
            currentCamera->manageEvent(mEvent);
        break;
    default:
        break;
    }
    delete mEvent;
    return true;
}

bool SofaViewer::mouseReleaseEvent_p( QMouseEvent * e)
{
    MouseEvent* mEvent = NULL;
    if (e->button() == Qt::LeftButton)
        mEvent = new MouseEvent(MouseEvent::LeftReleased,
                                e->x(), e->y());
    else if (e->button() == Qt::RightButton)
        mEvent = new MouseEvent(MouseEvent::RightReleased,
                                e->x(), e->y());
    else if (e->button() == Qt::MidButton)
        mEvent = new MouseEvent(MouseEvent::MiddleReleased,
                                e->x(), e->y());
    else{
        // A fallback event to rules them all...
        mEvent = new MouseEvent(MouseEvent::AnyExtraButtonReleased,
                                e->x(), e->y());
    }

    switch(m_state){
    case STATE_SCENEFORWARDING:
        if (m_simulationRoot)
            m_simulationRoot->propagateEvent(ExecParams::defaultInstance(), mEvent);
        break;
    case STATE_PICKING:
        updatePicking(e);
        break;
    case STATE_CAMERAMANIPULATION:
        if (currentCamera)
            currentCamera->manageEvent(mEvent);
        break;
    default:
        break;
    }

    delete mEvent;
    return true;
}

bool SofaViewer::updatePicking(QMouseEvent *e)
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);

    MousePosition mousepos;
    mousepos.screenWidth  = viewport[2];
    mousepos.screenHeight = viewport[3];
    mousepos.x      = e->x();
    mousepos.y      = e->y();

    getPickHandler()->activateRay(viewport[2],viewport[3], m_simulationRoot.get());
    getPickHandler()->updateMouse2D( mousepos );

    switch (e->type())
    {
    case QEvent::MouseButtonPress:

        if (e->button() == Qt::LeftButton)
        {
            getPickHandler()->handleMouseEvent(PRESSED, LEFT);
        }
        else if (e->button() == Qt::RightButton) // Shift+Rightclick to remove triangles
        {
            getPickHandler()->handleMouseEvent(PRESSED, RIGHT);
        }
        else if (e->button() == Qt::MidButton) // Shift+Midclick (by 2 steps defining 2 input points) to cut from one point to another
        {
            getPickHandler()->handleMouseEvent(PRESSED, MIDDLE);
        }
        break;
    case QEvent::MouseButtonRelease:
    {

        if (e->button() == Qt::LeftButton)
        {
            getPickHandler()->handleMouseEvent(RELEASED, LEFT);
        }
        else if (e->button() == Qt::RightButton)
        {
            getPickHandler()->handleMouseEvent(RELEASED, RIGHT);
        }
        else if (e->button() == Qt::MidButton)
        {
            getPickHandler()->handleMouseEvent(RELEASED, MIDDLE);
        }
    }
        break;
    default:
        break;
    }
    moveRayPickInteractor(e->x(), e->y());

    return true;
}

void SofaViewer::captureEvent()
{
    if (m_doVideoRecording)
    {
        bool skip = false;
        unsigned int frameskip = SofaVideoRecorderManager::getInstance()->getFrameskip();
        if (frameskip)
        {
            static unsigned int skipcounter = 0;
            if (skipcounter < frameskip)
            {
                skip = true;
                ++skipcounter;
            }
            else
            {
                skip = false;
                skipcounter = 0;
            }
        }
        if (!skip)
        {
            switch (SofaVideoRecorderManager::getInstance()->getRecordingType())
            {
            case SofaVideoRecorderManager::SCREENSHOTS :
                screenshot(capture.findFilename(), 1);
                break;
            case SofaVideoRecorderManager::MOVIE :
                #ifdef SOFA_HAVE_FFMPEG
                    videoRecorder.addFrame();
                #endif //
                break;
            default :
                break;
            }
        }
    }
}


}
}
}
}
