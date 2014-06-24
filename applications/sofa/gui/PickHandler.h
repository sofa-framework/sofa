/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_PICKHANDLER_H
#define SOFA_GUI_PICKHANDLER_H

#include "SofaGUI.h"
#include "OperationFactory.h"


#include <sofa/gui/ColourPickingVisitor.h>

#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/Node.h>

#include <SofaBaseMechanics/MechanicalObject.h>

#include <SofaUserInteraction/RayModel.h>
#include <SofaUserInteraction/MouseInteractor.h>

#include <SofaGraphComponent/MouseButtonSetting.h>

#include <sofa/helper/fixed_array.h>
#include <sofa/helper/gl/FrameBufferObject.h>
#include <functional>


namespace sofa
{
namespace component
{
namespace collision
{
class ComponentMouseInteraction;
}
}

namespace gui
{

using simulation::Node;
using sofa::component::collision::BodyPicked;
using sofa::component::collision::ComponentMouseInteraction;

class CallBackPicker
{
public:
    virtual ~CallBackPicker() {}
    virtual void execute(const sofa::component::collision::BodyPicked &body)=0;
};

class CallBackRender
{
public:
    virtual ~CallBackRender() {}
    virtual void render(ColourPickingVisitor::ColourCode code ) = 0;
};





class SOFA_SOFAGUI_API PickHandler
{
    typedef sofa::component::collision::RayModel MouseCollisionModel;
    typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > MouseContainer;

public:
    enum PickingMethod
    {
        RAY_CASTING,
        SELECTION_BUFFER
    };

    PickHandler();
    ~PickHandler();

    void activateRay(int width, int height, core::objectmodel::BaseNode* root);
    void deactivateRay();

    void allocateSelectionBuffer(int width, int height);
    void destroySelectionBuffer();


    void setPickingMethod(PickingMethod method) { pickingMethod = method; }
    bool useSelectionBufferMethod() const { return (pickingMethod == SELECTION_BUFFER); }

    void updateRay(const sofa::defaulttype::Vector3 &position, const sofa::defaulttype::Vector3 &orientation);

    void handleMouseEvent( MOUSE_STATUS status, MOUSE_BUTTON button);

    void init(core::objectmodel::BaseNode* root);
    void reset();
    void unload();

    void setColourRenderCallback(CallBackRender * colourRender)
    {
        renderCallback = colourRender;
    }

    void updateMouse2D( MousePosition mouse ) { mousePosition = mouse ;}


    Operation *getOperation(MOUSE_BUTTON button) {return operations[button];}

    Operation *changeOperation(sofa::component::configurationsetting::MouseButtonSetting* setting);
    Operation *changeOperation(MOUSE_BUTTON button, const std::string &op);

    void addCallBack(CallBackPicker *c) {callbacks.push_back(c);}
    helper::vector< CallBackPicker* > getCallBackPicker() {return callbacks;}
    void clearCallBacks() {for (unsigned int i=0; i<callbacks.size(); ++i) callbacks.clear();}

    static BodyPicked findCollisionUsingBruteForce(const defaulttype::Vector3& origin, const defaulttype::Vector3& direction, double maxLength, core::objectmodel::BaseNode* root);
    BodyPicked findCollisionUsingColourCoding(const defaulttype::Vector3& origin, const defaulttype::Vector3& direction);

    ComponentMouseInteraction           *getInteraction();
    BodyPicked                          *getLastPicked() {return &lastPicked;}

protected:
    bool interactorInUse;
    MOUSE_STATUS mouseStatus;
    MOUSE_BUTTON mouseButton;


    Node::SPtr                mouseNode;
    MouseContainer::SPtr      mouseContainer;
    MouseCollisionModel::SPtr mouseCollision;

    MousePosition             mousePosition;

#ifndef SOFA_NO_OPENGL
    sofa::helper::gl::FrameBufferObject _fbo;
    sofa::helper::gl::fboParameters     _fboParams;
#endif

    ComponentMouseInteraction *interaction;
    std::vector< ComponentMouseInteraction *> instanceComponents;


    BodyPicked lastPicked;

    bool useCollisions;



    //NONE is the number of Operations in use.
    helper::fixed_array< Operation*,NONE > operations;

    helper::vector< CallBackPicker* > callbacks;

    CallBackRender* renderCallback;

    PickingMethod pickingMethod;

    bool _fboAllocated;


    BodyPicked findCollision();
    BodyPicked findCollisionUsingPipeline();
    BodyPicked findCollisionUsingBruteForce();
    BodyPicked findCollisionUsingColourCoding();

    bool needToCastRay();
    void setCompatibleInteractor();


};
}
}

#endif
