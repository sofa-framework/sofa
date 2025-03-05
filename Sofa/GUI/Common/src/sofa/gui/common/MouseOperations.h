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

#include <sofa/helper/cast.h>

#include <sofa/component/setting/MouseButtonSetting.h>
#include <sofa/gui/component/AddRecordedCameraButtonSetting.h>
#include <sofa/gui/component/AttachBodyButtonSetting.h>
#include <sofa/gui/component/FixPickedParticleButtonSetting.h>

#include <iostream>
#include <vector>

namespace sofa::gui::component::performer
{
    class InteractionPerformer;
} // namespace sofa::gui::component::performer

namespace sofa::gui::common
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

class SOFA_GUI_COMMON_API Operation
{
    friend class OperationFactory;
public:
    Operation(sofa::component::setting::MouseButtonSetting::SPtr s = nullptr): pickHandle(nullptr),mbsetting(s),performer(nullptr),button(NONE) {}
    virtual ~Operation() {}
    virtual void configure(PickHandler*picker, MOUSE_BUTTON b) { pickHandle=picker; button=b; }
    virtual void configure(PickHandler* picker, sofa::component::setting::MouseButtonSetting* s)
    { setSetting(s); configure(picker,GetMouseId(s->d_button.getValue().getSelectedId())); }
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
    sofa::component::setting::MouseButtonSetting::SPtr mbsetting;
public:
    virtual void setSetting(sofa::component::setting::MouseButtonSetting* s) { mbsetting = s; }
    sofa::gui::component::performer::InteractionPerformer *performer;
    virtual std::string defaultPerformerType() { return ""; }
    virtual sofa::gui::component::performer::InteractionPerformer *createPerformer();
    virtual void configurePerformer(sofa::gui::component::performer::InteractionPerformer* p);
    MOUSE_BUTTON getMouseButton() const { return button; }
    std::string getId() { return id; }
protected:
    MOUSE_BUTTON button;
private:
    std::string id;
};

class SOFA_GUI_COMMON_API AttachOperation : public Operation
{
public:
    AttachOperation(sofa::gui::component::AttachBodyButtonSetting::SPtr s = sofa::core::objectmodel::New<sofa::gui::component::AttachBodyButtonSetting>()) : Operation(s), setting(s)
    {}
    ~AttachOperation() override {}

    void setStiffness(double s) {setting->d_stiffness.setValue(s);}
    double getStiffness() const { return setting->d_stiffness.getValue();}
    void setArrowSize(double s) {setting->d_arrowSize.setValue(s);}
    double getArrowSize() const { return setting->d_arrowSize.getValue();}
    void setShowFactorSize(double s) { setting->d_showFactorSize.setValue(s); }
    double getShowFactorSize() const { return setting->d_showFactorSize.getValue(); }

    static std::string getDescription() {return "Attach an object to the Mouse using a spring force field";}

protected:
    void setSetting(sofa::component::setting::MouseButtonSetting* s) override { Operation::setSetting(s); setting = down_cast<sofa::gui::component::AttachBodyButtonSetting>(s); }
    virtual std::string defaultPerformerType() override;
    void configurePerformer(sofa::gui::component::performer::InteractionPerformer* p) override;

    sofa::gui::component::AttachBodyButtonSetting::SPtr setting;
};


class SOFA_GUI_COMMON_API ConstraintAttachOperation : public sofa::gui::common::AttachOperation
{
public:
    static std::string getDescription() { return "Attach an object to the mouse using a bilateral interaction constraint"; }
protected:
    virtual std::string defaultPerformerType() { return "ConstraintAttachBody"; }
};

class SOFA_GUI_COMMON_API FixOperation : public Operation
{
public:
    FixOperation() : setting(sofa::core::objectmodel::New<sofa::gui::component::FixPickedParticleButtonSetting>())
    {}

    ~FixOperation() override {}

    void setStiffness(double s) {setting->stiffness.setValue(s); }
    virtual double getStiffness() const { return setting->stiffness.getValue();}

    static std::string getDescription() {return "Fix Picked particle";}
protected:
    virtual std::string defaultPerformerType() override;
    void configurePerformer(sofa::gui::component::performer::InteractionPerformer* p) override;

    sofa::gui::component::FixPickedParticleButtonSetting::SPtr setting;
};

class SOFA_GUI_COMMON_API AddFrameOperation : public Operation
{
public:
    static std::string getDescription() {return "Add a Frame to a Skinned model";}
protected:
    virtual std::string defaultPerformerType() override;
    void configurePerformer(sofa::gui::component::performer::InteractionPerformer* p) override;
};

class SOFA_GUI_COMMON_API AddRecordedCameraOperation : public Operation
{
public:
	AddRecordedCameraOperation() : setting(sofa::core::objectmodel::New<sofa::gui::component::AddRecordedCameraButtonSetting>())
	{}
	static std::string getDescription() {return "Save camera's view points for navigation ";}
protected:
    virtual std::string defaultPerformerType() override;
	void configurePerformer(sofa::gui::component::performer::InteractionPerformer* p) override;
	sofa::gui::component::AddRecordedCameraButtonSetting::SPtr setting;
};

class SOFA_GUI_COMMON_API StartNavigationOperation : public Operation
{
public:
	StartNavigationOperation() : setting(sofa::core::objectmodel::New<sofa::gui::component::StartNavigationButtonSetting>())
	{}
	static std::string getDescription() {return "Start navigation if camera's view points have been saved";}
protected:
    virtual std::string defaultPerformerType() override;
	void configurePerformer(sofa::gui::component::performer::InteractionPerformer* p) override;
	sofa::gui::component::StartNavigationButtonSetting::SPtr setting;
};

class SOFA_GUI_COMMON_API InciseOperation : public Operation
{
public:
    InciseOperation():startPerformer(nullptr), cpt (0) {}
    ~InciseOperation() override;
    void start() override ;
    void execution() override ;
    void end() override ;
    void endOperation() override ;

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
    sofa::gui::component::performer::InteractionPerformer *startPerformer;

    int method;
    int snapingBorderValue;
    int snapingValue;
    int cpt;
    bool finishIncision;
    bool keepPoint;
};

class SOFA_GUI_COMMON_API TopologyOperation : public Operation
{
public:
    TopologyOperation():scale (0.0), volumicMesh (false), firstClick(true) {}

    ~TopologyOperation() override {}
    void start() override;
    void execution() override;
    void end() override;
    void endOperation() override;

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


class SOFA_GUI_COMMON_API AddSutureOperation : public Operation
{
public:
    AddSutureOperation():stiffness(10.0), damping(1.0) {}
    ~AddSutureOperation() override {}

    void setStiffness(double f) { stiffness = f;}
    virtual double getStiffness() const {return stiffness;}
    void setDamping(double f) {damping = f;}
    virtual double getDamping() const {return damping;}

    static std::string getDescription() {return "Add a spring to suture two points.";}
protected:
    virtual std::string defaultPerformerType() override;
    void configurePerformer(sofa::gui::component::performer::InteractionPerformer* p) override;

    double stiffness;
    double damping;
};

} // namespace sofa::gui::common
