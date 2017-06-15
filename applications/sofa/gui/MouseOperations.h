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
#ifndef SOFA_GUI_MOUSEOPERATIONS_H
#define SOFA_GUI_MOUSEOPERATIONS_H

#include "SofaGUI.h"
#include <iostream>
#include <vector>

#include <SofaGraphComponent/AddRecordedCameraButtonSetting.h>
#include <SofaGraphComponent/AttachBodyButtonSetting.h>
#include <SofaGraphComponent/FixPickedParticleButtonSetting.h>
#include <sofa/gui/SofaGUI.h>
#include <sofa/helper/cast.h>

namespace sofa
{
namespace component
{
namespace collision
{
class InteractionPerformer;
}
}
namespace gui
{
enum MOUSE_BUTTON {LEFT, MIDDLE, RIGHT,NONE};
enum MOUSE_STATUS {PRESSED,RELEASED, ACTIVATED, DEACTIVATED};


struct MousePosition
{
    int x;
    int y;
    int screenWidth;
    int screenHeight;
};

class PickHandler;

class SOFA_SOFAGUI_API Operation
{
    friend class OperationFactory;
public:
    Operation(sofa::component::configurationsetting::MouseButtonSetting::SPtr s = NULL): pickHandle(NULL),mbsetting(s),performer(NULL),button(NONE) {}
    virtual ~Operation() {}
    virtual void configure(PickHandler*picker, MOUSE_BUTTON b) { pickHandle=picker; button=b; }
    virtual void configure(PickHandler* picker, sofa::component::configurationsetting::MouseButtonSetting* s)
    { setSetting(s); configure(picker,GetMouseId(s->button.getValue().getSelectedId())); }
    virtual void start();                      /// This function is called each time the mouse is clicked.
    virtual void execution() {}
    virtual void end();                        /// This function is called after each mouse click.
    virtual void endOperation() { this->end(); }  /// This function is called when shift key is released.
    virtual void wait() {}
    static MOUSE_BUTTON GetMouseId(unsigned int i)
    {
        switch (i)
        {
        case LEFT:   return LEFT;
        case MIDDLE: return MIDDLE;
        case RIGHT:  return RIGHT;
        default:     return NONE;
        }
    }
protected:
    PickHandler *pickHandle;
    sofa::component::configurationsetting::MouseButtonSetting::SPtr mbsetting;
public:
    virtual void setSetting(sofa::component::configurationsetting::MouseButtonSetting* s) { mbsetting = s; }
    sofa::component::collision::InteractionPerformer *performer;
    virtual std::string defaultPerformerType() { return ""; }
    virtual sofa::component::collision::InteractionPerformer *createPerformer();
    virtual void configurePerformer(sofa::component::collision::InteractionPerformer* p);
    MOUSE_BUTTON getMouseButton() const { return button; }
    std::string getId() { return id; }
protected:
    MOUSE_BUTTON button;
private:
    std::string id;
};

class SOFA_SOFAGUI_API AttachOperation : public Operation
{
public:
    AttachOperation(sofa::component::configurationsetting::AttachBodyButtonSetting::SPtr s = sofa::core::objectmodel::New<sofa::component::configurationsetting::AttachBodyButtonSetting>()) : Operation(s), setting(s)
    {}
    virtual ~AttachOperation() {}

    void setStiffness(double s) {setting->stiffness.setValue(s);}
    double getStiffness() const { return setting->stiffness.getValue();}
    void setArrowSize(double s) {setting->arrowSize.setValue(s);}
    double getArrowSize() const { return setting->arrowSize.getValue();}
    void setShowFactorSize(double s) { setting->showFactorSize.setValue(s); }
    double getShowFactorSize() const { return setting->showFactorSize.getValue(); }

    static std::string getDescription() {return "Attach an object to the Mouse";}

protected:
    virtual void setSetting(sofa::component::configurationsetting::MouseButtonSetting* s) { Operation::setSetting(s); setting = down_cast<sofa::component::configurationsetting::AttachBodyButtonSetting>(s); }
    virtual std::string defaultPerformerType();
    virtual void configurePerformer(sofa::component::collision::InteractionPerformer* p);

    sofa::component::configurationsetting::AttachBodyButtonSetting::SPtr setting;
};


class SOFA_SOFAGUI_API ConstraintAttachOperation : public sofa::gui::AttachOperation
{
public:
    static std::string getDescription() { return "Attach an object to the mouse using a bilateral interaction constraint"; }
protected:
    virtual std::string defaultPerformerType() { return "ConstraintAttachBody"; }
};

class SOFA_SOFAGUI_API FixOperation : public Operation
{
public:
    FixOperation() : setting(sofa::core::objectmodel::New<sofa::component::configurationsetting::FixPickedParticleButtonSetting>())
    {}

    virtual ~FixOperation() {};

    void setStiffness(double s) {setting->stiffness.setValue(s); }
    virtual double getStiffness() const { return setting->stiffness.getValue();}

    static std::string getDescription() {return "Fix Picked particle";}
protected:
    virtual std::string defaultPerformerType();
    virtual void configurePerformer(sofa::component::collision::InteractionPerformer* p);

    sofa::component::configurationsetting::FixPickedParticleButtonSetting::SPtr setting;
};

class SOFA_SOFAGUI_API AddFrameOperation : public Operation
{
public:
    static std::string getDescription() {return "Add a Frame to a Skinned model";}
protected:
    virtual std::string defaultPerformerType();
    virtual void configurePerformer(sofa::component::collision::InteractionPerformer* p);
};

class SOFA_SOFAGUI_API AddRecordedCameraOperation : public Operation
{
public:
	AddRecordedCameraOperation() : setting(sofa::core::objectmodel::New<sofa::component::configurationsetting::AddRecordedCameraButtonSetting>())
	{}
	static std::string getDescription() {return "Save camera's view points for navigation ";}
protected:
	virtual std::string defaultPerformerType();
	virtual void configurePerformer(sofa::component::collision::InteractionPerformer* p);
	sofa::component::configurationsetting::AddRecordedCameraButtonSetting::SPtr setting;
};

class SOFA_SOFAGUI_API StartNavigationOperation : public Operation
{
public:
	StartNavigationOperation() : setting(sofa::core::objectmodel::New<sofa::component::configurationsetting::StartNavigationButtonSetting>())
	{}
	static std::string getDescription() {return "Start navigation if camera's view points have been saved";}
protected:
	virtual std::string defaultPerformerType();
	virtual void configurePerformer(sofa::component::collision::InteractionPerformer* p);
	sofa::component::configurationsetting::StartNavigationButtonSetting::SPtr setting;
};

class SOFA_SOFAGUI_API InciseOperation : public Operation
{
public:
    InciseOperation():startPerformer(NULL), cpt (0) {};
    virtual ~InciseOperation();
    virtual void start() ;
    virtual void execution() ;
    virtual void end() ;
    virtual void endOperation() ;

    void setIncisionMethod (int m) {method = m;}
    void setSnapingBorderValue (int m) {snapingBorderValue = m;}
    void setSnapingValue (int m) {snapingValue = m;}
    void setCompleteIncision (bool m) {finishIncision = m;}
    void setKeepPoint (bool m) {keepPoint = m;}

    virtual int getIncisionMethod() const { return method;}
    virtual int getSnapingBorderValue() const { return snapingBorderValue;}
    virtual int getSnapingValue() const { return snapingValue;}
    virtual bool getCompleteIncision() {return finishIncision;}
    virtual bool getKeepPoint() {return keepPoint;}

    static std::string getDescription() {return "Incise along a path";}
protected:
    sofa::component::collision::InteractionPerformer *startPerformer;

    int method;
    int snapingBorderValue;
    int snapingValue;
    int cpt;
    bool finishIncision;
    bool keepPoint;
};

class SOFA_SOFAGUI_API TopologyOperation : public Operation
{
public:
    TopologyOperation():scale (0.0), volumicMesh (0), firstClick(1) {};

    virtual ~TopologyOperation() {};
    virtual void start();
    virtual void execution();
    virtual void end();
    virtual void endOperation();

    void setTopologicalOperation(int m) {topologicalOperation = m;}
    void setScale (double s) {scale = s;}
    void setVolumicMesh (bool v) {volumicMesh = v;}

    virtual int getTopologicalOperation() const { return volumicMesh;}
    virtual double getScale() const {return scale;}
    virtual bool getVolumicMesh() const {return volumicMesh;}

    static std::string getDescription() {return "Perform topological operations";}

protected:
    int topologicalOperation;
    double scale;
    bool volumicMesh;
    bool firstClick;
};


class SOFA_SOFAGUI_API AddSutureOperation : public Operation
{
public:
    AddSutureOperation():stiffness(10.0), damping(1.0) {}
    virtual ~AddSutureOperation() {}

    void setStiffness(double f) { stiffness = f;}
    virtual double getStiffness() const {return stiffness;}
    void setDamping(double f) {damping = f;}
    virtual double getDamping() const {return damping;}

    static std::string getDescription() {return "Add a spring to suture two points.";}
protected:
    virtual std::string defaultPerformerType();
    virtual void configurePerformer(sofa::component::collision::InteractionPerformer* p);

    double stiffness;
    double damping;
};

}
}

#endif
