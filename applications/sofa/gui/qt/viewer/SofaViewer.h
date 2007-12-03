/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This program is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU General Public License as published by the Free   *
* Software Foundation; either version 2 of the License, or (at your option)    *
* any later version.                                                           *
*                                                                              *
* This program is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for     *
* more details.                                                                *
*                                                                              *
* You should have received a copy of the GNU General Public License along with *
* this program; if not, write to the Free Software Foundation, Inc., 51        *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                    *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_VIEWER_H
#define SOFA_VIEWER_H

#include <qstring.h>
#include <qwidget.h>



#ifdef QT_MODULE_QT3SUPPORT
#include <QEvent>
#include <QMouseEvent>
#include <QKeyEvent>
#else
#include <qevent.h>
#endif

#include <sofa/simulation/automatescheduler/Automate.h>
#include <sofa/helper/gl/Capture.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/component/collision/RayPickInteractor.h>

//instruments handling
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/simulation/tree/GrabVisitor.h>
#include <sofa/simulation/tree/MechanicalVisitor.h>
#include <sofa/simulation/tree/UpdateMappingVisitor.h>
#include <sofa/simulation/tree/Simulation.h>
#ifdef QT_MODULE_QT3SUPPORT
#include <QEvent>
#include <QMouseEvent>
#include <QKeyEvent>
#else
#include <qevent.h>
#endif

namespace sofa
{

namespace gui
{

namespace qt
{

namespace viewer
{

using namespace sofa::defaulttype;
enum
{
    BTLEFT_MODE = 101,
    BTRIGHT_MODE = 102,
    BTMIDDLE_MODE = 103,
};

enum {CAMERA_PERSPECTIVE, CAMERA_ORTHOGRAPHIC};
//      using namespace sofa::simulation::automatescheduler;

class SofaViewer : public sofa::simulation::automatescheduler::Automate::DrawCB
{

public:
    SofaViewer ()
        : groot(NULL), m_isControlPressed(false), _video(false), _shadow(false), _axis(false), camera_type(CAMERA_PERSPECTIVE)
    {}
    ~SofaViewer() {}

    virtual QWidget* getQWidget()=0;

    virtual sofa::simulation::tree::GNode* getScene()        {  return groot;}
    virtual const std::string&             getSceneFileName() {  return sceneFileName;}
    virtual void                           setSceneFileName(const std::string &f) {sceneFileName = f;};

    virtual void setup() {}
    virtual void setScene(sofa::simulation::tree::GNode* scene, const char* filename=NULL, bool keepParams=false)=0;
    virtual void resetView()=0;
    virtual QString helpString()=0;
    virtual bool ready() { return true; }
    virtual void wait() {}

    //Fonctions needed to take a screenshot
    virtual const std::string screenshotName() { return capture.findFilename().c_str();};
    virtual void        setPrefix(const std::string filename) {capture.setPrefix(filename);};
    virtual void        screenshot(const std::string filename)=0;

protected:



    // ---------------------- Here are the Keyboard controls   ----------------------
    void keyPressEvent ( QKeyEvent * e )
    {
        switch(e->key())
        {
        case Qt::Key_B:
            // --- change background
        {
            _background = (_background+1)%3;
            break;
        }
        case Qt::Key_L:
            // --- draw shadows
        {
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
        case Qt::Key_T:
        {
            if (camera_type == CAMERA_PERSPECTIVE) camera_type = CAMERA_ORTHOGRAPHIC;
            else                                   camera_type = CAMERA_PERSPECTIVE;
            break;
        }
        case Qt::Key_V:
            // --- save video
        {
            _video = !_video;
            capture.setCounter();
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
        case Qt::Key_Escape:
        {
            exit(0);
            break;
        }
        default:
        {
            e->ignore();
        }
        }
    }


    void keyReleaseEvent ( QKeyEvent * e )
    {
        switch(e->key())
        {
        case Qt::Key_Control:
        {
            m_isControlPressed = false;
        }
        default:
        {
            e->ignore();
        }
        }
        if( isControlPressed() )
        {
            sofa::core::objectmodel::KeyreleasedEvent keyEvent(e->key());
            if (groot) groot->propagateEvent(&keyEvent);
        }
    }




    bool isControlPressed() const
    {
        return m_isControlPressed;
    }



    // ---------------------- Here are the Mouse controls   ----------------------

    void mouseEvent( QMouseEvent *e)
    {
        if (e->state()&Qt::ShiftButton)
        {
            //_sceneTransform.ApplyInverse();
            if (interactor==NULL)
            {
                interactor = new sofa::component::collision::RayPickInteractor();
                interactor->setName("mouse");
                if (groot)
                {
                    simulation::tree::GNode* child = new simulation::tree::GNode("mouse");
                    groot->addChild(child);
                    child->addObject(interactor);
                }
            }
            interactor->newEvent("show");
            switch (e->type())
            {
            case QEvent::MouseButtonPress:
                if (e->button() == Qt::LeftButton)
                {
                    interactor->newEvent("pick"); // Shift+Leftclick to deform the mesh
                }
                else if (e->button() == Qt::RightButton) // Shift+Rightclick to remove triangles
                {
                    interactor->newEvent("pick2");
                }
                else if (e->button() == Qt::MidButton) // Shift+Midclick (by 2 steps defining 2 input points) to cut from one point to another
                {
                    interactor->newEvent("pick3");
                }
                break;
            case QEvent::MouseButtonRelease:
                //if (e->button() == Qt::LeftButton)
            {
                interactor->newEvent("release");
            }
            break;
            default: break;
            }
            moveRayPickInteractor(e->x(), e->y());
        }

    }

    // ---------------------- Here are the controls for instruments  ----------------------

    void moveLaparoscopic( QMouseEvent *e)
    {

        int index_instrument = simulation::tree::getSimulation()->instrumentInUse.getValue();
        if (index_instrument < 0 || index_instrument > (int)simulation::tree::getSimulation()->instruments.size()) return;

        simulation::tree::GNode *instrument = simulation::tree::getSimulation()->instruments[index_instrument];
        if (instrument == NULL) return;

        int eventX = e->x();
        int eventY = e->y();

        std::vector< sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* > instruments;
        instrument->getTreeObjects<sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>, std::vector< sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* > >(&instruments);

        if (!instruments.empty())
        {
            sofa::core::componentmodel::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* instrument = instruments[0];
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
                        static_cast<sofa::simulation::tree::GNode*>(instrument->getContext())->execute<sofa::simulation::tree::GrabVisitor>();
                    }
                }
                break;

            default:
                break;
            }
            if (_mouseInteractorMoving && _navigationMode == BTLEFT_MODE)
            {
                int dx = eventX - _mouseInteractorSavedPosX;
                int dy = eventY - _mouseInteractorSavedPosY;
                if (dx || dy)
                {
                    (*instrument->getX())[0].getOrientation() = (*instrument->getX())[0].getOrientation() * Quat(Vector3(0,1,0),dx*0.001) * Quat(Vector3(0,0,1),dy*0.001);

                    _mouseInteractorSavedPosX = eventX;
                    _mouseInteractorSavedPosY = eventY;
                }
            }
            else if (_mouseInteractorMoving && _navigationMode == BTMIDDLE_MODE)
            {
                int dx = eventX - _mouseInteractorSavedPosX;
                int dy = eventY - _mouseInteractorSavedPosY;
                if (dx || dy)
                {
                    _mouseInteractorSavedPosX = eventX;
                    _mouseInteractorSavedPosY = eventY;
                }
            }
            else if (_mouseInteractorMoving && _navigationMode == BTRIGHT_MODE)
            {
                int dx = eventX - _mouseInteractorSavedPosX;
                int dy = eventY - _mouseInteractorSavedPosY;
                if (dx || dy)
                {
                    (*instrument->getX())[0].getTranslation() += (dy)*0.01;
                    (*instrument->getX())[0].getOrientation() = (*instrument->getX())[0].getOrientation() * Quat(Vector3(1,0,0),dx*0.001);
                    _mouseInteractorSavedPosX = eventX;
                    _mouseInteractorSavedPosY = eventY;
                }
            }
            static_cast<sofa::simulation::tree::GNode*>(instrument->getContext())->execute<sofa::simulation::tree::MechanicalPropagatePositionAndVelocityVisitor>();
            static_cast<sofa::simulation::tree::GNode*>(instrument->getContext())->execute<sofa::simulation::tree::UpdateMappingVisitor>();
            getQWidget()->update();
        }
    }

    virtual void moveRayPickInteractor(int , int ) {};

    sofa::helper::gl::Capture capture;
    sofa::simulation::tree::GNode* groot;
    std::string sceneFileName;

    bool m_isControlPressed;
    bool _video;
    bool _shadow;
    bool _axis;
    int  camera_type;
    int _background;
    sofa::component::collision::RayPickInteractor* interactor;

    //instruments handling
    int	_navigationMode;
    bool _mouseInteractorMoving;
    int _mouseInteractorSavedPosX;
    int _mouseInteractorSavedPosY;

    //*************************************************************
    // QT
    //*************************************************************
    //SLOTS
    virtual void saveView()=0;
    virtual void setSizeW(int)=0;
    virtual void setSizeH(int)=0;


    //SIGNALS
    virtual void redrawn()=0;
    virtual void resizeW( int )=0;
    virtual void resizeH( int )=0;

};
}
}
}
}

#endif
