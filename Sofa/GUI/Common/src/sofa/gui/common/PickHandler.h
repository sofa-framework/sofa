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
#pragma once
#include <sofa/gui/common/config.h>
#include <sofa/gui/common/OperationFactory.h>

#include <sofa/simulation/fwd.h>
#include <sofa/simulation/Node.h>

#include <sofa/gui/common/ColourPickingVisitor.h>
#include <sofa/component/statecontainer/MechanicalObject.h>

namespace sofa::component::collision::geometry
{
    class RayCollisionModel;
} // namespace sofa::component::collision::geometry

namespace sofa::gui::component::performer
{
    class ComponentMouseInteraction;
} // namespace sofa::gui::component::performer

namespace sofa::component::setting
{
    class MouseButtonSetting;
} // namespace sofa::component::setting


namespace sofa::gui::common
{

using simulation::Node;
using sofa::gui::component::performer::BodyPicked;
using sofa::gui::component::performer::ComponentMouseInteraction;

class CallBackPicker
{
public:
    virtual ~CallBackPicker() {}
    virtual void execute(const BodyPicked &body)=0;
};

class CallBackRender
{
public:
    virtual ~CallBackRender() {}
    virtual void render(ColourPickingVisitor::ColourCode code ) = 0;
};

class SOFA_GUI_COMMON_API PickHandler
{
    typedef sofa::component::collision::geometry::RayCollisionModel MouseCollisionModel;
    typedef sofa::component::statecontainer::MechanicalObject< defaulttype::Vec3Types > MouseContainer;

public:
    enum PickingMethod
    {
        RAY_CASTING,
        SELECTION_BUFFER
    };

    PickHandler(double defaultLength = 1000000);
    virtual ~PickHandler();

    void activateRay(int width, int height, core::objectmodel::BaseNode* root);
    void deactivateRay();

    virtual void allocateSelectionBuffer(int width, int height);
    virtual void destroySelectionBuffer();

    void setPickingMethod(PickingMethod method) { pickingMethod = method; }
    bool useSelectionBufferMethod() const { return (pickingMethod == SELECTION_BUFFER); }

    void updateRay(const sofa::type::Vec3 &position, const sofa::type::Vec3 &orientation);

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

    Operation *changeOperation(sofa::component::setting::MouseButtonSetting* setting);
    Operation *changeOperation(MOUSE_BUTTON button, const std::string &op);

    void addCallBack(CallBackPicker *c) {callbacks.push_back(c);}
    type::vector< CallBackPicker* > getCallBackPicker() {return callbacks;}
    void clearCallBacks() {for (unsigned int i=0; i<callbacks.size(); ++i) callbacks.clear();}

    static BodyPicked findCollisionUsingBruteForce(const type::Vec3& origin, const type::Vec3& direction, double maxLength, core::objectmodel::BaseNode* root);
    virtual BodyPicked findCollisionUsingColourCoding(const type::Vec3& origin, const type::Vec3& direction);

    ComponentMouseInteraction           *getInteraction();
    BodyPicked                          *getLastPicked() {return &lastPicked;}

protected:
    bool interactorInUse;
    MOUSE_STATUS mouseStatus;
    MOUSE_BUTTON mouseButton;


    sofa::simulation::NodeSPtr     mouseNode;
    MouseContainer::SPtr      mouseContainer;
    sofa::core::sptr<MouseCollisionModel> mouseCollision;

    MousePosition             mousePosition;

    ComponentMouseInteraction *interaction;
    std::vector< ComponentMouseInteraction *> instanceComponents;


    BodyPicked lastPicked;

    bool useCollisions;



    //NONE is the number of Operations in use.
    type::fixed_array< Operation*,NONE > operations;

    type::vector< CallBackPicker* > callbacks;

    CallBackRender* renderCallback;

    PickingMethod pickingMethod;


    virtual BodyPicked findCollision();
    BodyPicked findCollisionUsingPipeline();
    BodyPicked findCollisionUsingBruteForce();
    BodyPicked findCollisionUsingColourCoding();

    bool needToCastRay();
    void setCompatibleInteractor();

    /// Default length of the ray. Set by constructor.
    double m_defaultLength;
};

} // namespace sofa::gui::common
