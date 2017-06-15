/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "SofaViewer.h"
#include <sofa/helper/Factory.inl>
#include <SofaBaseVisual/VisualStyle.h>
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
    : sofa::gui::BaseViewer()
    , m_isControlPressed(false)
{
    colourPickingRenderCallBack = ColourPickingRenderCallBack(this);
}

SofaViewer::~SofaViewer()
{
}

void SofaViewer::redraw()
{
    getQWidget()->update();
}

void SofaViewer::keyPressEvent(QKeyEvent * e)
{
    sofa::core::objectmodel::KeypressedEvent kpe(e->key());
    if(currentCamera)
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
        getPickHandler()->activateRay(viewport[2],viewport[3], groot.get());
        break;
    case Qt::Key_B:
        // --- change background
    {
        _background = (_background + 1) % 3;
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
                unsigned int bitrate = videoManager->getBitrate();
                unsigned int framerate = videoManager->getFramerate();
                std::string videoFilename = videoRecorder.findFilename(videoManager->getCodecExtension());
                videoRecorder.init( videoFilename, framerate, bitrate, videoManager->getCodecName());
#endif

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
        case sofa::component::visualmodel::BaseCamera::PARALLEL:
            currentCamera->setStereoStrategy(sofa::component::visualmodel::BaseCamera::TOEDIN);
            std::cout << "Stereo Strategy: TOEDIN" << std::endl;
            break;
        case sofa::component::visualmodel::BaseCamera::TOEDIN:
            currentCamera->setStereoStrategy(sofa::component::visualmodel::BaseCamera::PARALLEL);
            std::cout << "Stereo Strategy: Parallel" << std::endl;
            break;
        default:
            currentCamera->setStereoStrategy(sofa::component::visualmodel::BaseCamera::PARALLEL);
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
        case sofa::component::visualmodel::BaseCamera::STEREO_INTERLACED:
            std::cout << "Stereo mode: Interlaced" << std::endl;
            break;
        case sofa::component::visualmodel::BaseCamera::STEREO_SIDE_BY_SIDE:
            std::cout << "Stereo mode: Side by Side" << std::endl; break;
        case sofa::component::visualmodel::BaseCamera::STEREO_SIDE_BY_SIDE_HALF:
            std::cout << "Stereo mode: Side by Side Half" << std::endl; break;
        case sofa::component::visualmodel::BaseCamera::STEREO_FRAME_PACKING:
            std::cout << "Stereo mode: Frame Packing" << std::endl; break;
        case sofa::component::visualmodel::BaseCamera::STEREO_TOP_BOTTOM:
            std::cout << "Stereo mode: Top Bottom" << std::endl; break;
        case sofa::component::visualmodel::BaseCamera::STEREO_TOP_BOTTOM_HALF:
            std::cout << "Stereo mode: Top Bottom Half" << std::endl; break;
        case sofa::component::visualmodel::BaseCamera::STEREO_AUTO:
            std::cout << "Stereo mode: Automatic" << std::endl; break;
        case sofa::component::visualmodel::BaseCamera::STEREO_NONE:
            std::cout << "Stereo mode: None" << std::endl; break;
        default:
            std::cout << "Stereo mode: INVALID" << std::endl; break;
            break;
        }
        break;
    }
    case Qt::Key_Control:
    {
        m_isControlPressed = true;
//        msg_info()<<"QtViewer::keyPressEvent, CONTROL pressed"<<std::endl;
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
        getPickHandler()->deactivateRay();

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
	if (!currentCamera) return;
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
	if (!currentCamera) return;
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
	if (!currentCamera) return;
    sofa::core::objectmodel::MouseEvent* mEvent = NULL;
    if (e->button() == Qt::LeftButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftPressed, e->x(), e->y());
    else if (e->button() == Qt::RightButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightPressed, e->x(), e->y());
    else if (e->button() == Qt::MidButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddlePressed, e->x(), e->y());
	else{
		// A fallback event to rules them all... 
	    mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::AnyExtraButtonPressed, e->x(), e->y());
	}
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
	if (!currentCamera) return;
    sofa::core::objectmodel::MouseEvent* mEvent = NULL;
    if (e->button() == Qt::LeftButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftReleased, e->x(), e->y());
    else if (e->button() == Qt::RightButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightReleased, e->x(), e->y());
    else if (e->button() == Qt::MidButton)
        mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddleReleased, e->x(), e->y());
	else{
		// A fallback event to rules them all... 
	    mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::AnyExtraButtonReleased, e->x(), e->y());
	}

    currentCamera->manageEvent(mEvent);

    getQWidget()->update();
#ifndef SOFA_GUI_INTERACTION
    if (groot)
        groot->propagateEvent(core::ExecParams::defaultInstance(), mEvent);
#endif
}

bool SofaViewer::mouseEvent(QMouseEvent *e)
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);

    MousePosition mousepos;
    mousepos.screenWidth  = viewport[2];
    mousepos.screenHeight = viewport[3];
    mousepos.x      = e->x();
    mousepos.y      = e->y();

    if (e->modifiers() & Qt::ShiftModifier)
    {

        getPickHandler()->activateRay(viewport[2],viewport[3], groot.get());
        getPickHandler()->updateMouse2D( mousepos );

        //_sceneTransform.ApplyInverse();
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
            //if (e->button() == Qt::LeftButton)
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
    }
    else
    {
        getPickHandler()->activateRay(viewport[2],viewport[3], groot.get());
    }
    return true;
}

void SofaViewer::captureEvent()
{
    if (_video)
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
#endif //SOFA_HAVE_FFMPEG
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
