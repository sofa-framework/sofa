/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_MOUSEOPERATIONS_H
#define SOFA_GUI_MOUSEOPERATIONS_H

#include "SofaGUI.h"
#include <iostream>
#include <vector>

#include <sofa/component/configurationsetting/AttachBodyButtonSetting.h>

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

class PickHandler;

class Operation
{
public:
    Operation(): pickHandle(NULL), performer(NULL),button(NONE) {};
    virtual ~Operation() {};
    virtual void configure(PickHandler *picker, MOUSE_BUTTON b) {pickHandle=picker; button=b; }
    virtual void start() =0;                   /// This function is called each time the mouse is clicked.
    virtual void execution() =0;
    virtual void end()     =0;                 /// This function is called after each mouse click.
    virtual void endOperation() {this->end();}; /// This function is called when shift key is released.
    virtual void wait() {};
protected:
    PickHandler *pickHandle;
public:
    sofa::component::collision::InteractionPerformer *performer;
protected:
    MOUSE_BUTTON button;
};

class SOFA_SOFAGUI_API AttachOperation : public Operation
{
public:
    AttachOperation();
    virtual ~AttachOperation() {};
    virtual void start() ;
    virtual void execution() ;
    virtual void end() ;
    virtual void endOperation() ;

    void setStiffness(double s) {setting.setStiffness(s);}
    virtual double getStiffness() const { return setting.getStiffness();}
    void setArrowSize(double s) {setting.setArrowSize(s);}
    virtual double getArrowSize() const { return setting.getArrowSize();}

    static std::string getDescription() {return "Attach an object to the Mouse";}
protected:
    sofa::component::configurationsetting::AttachBodyButtonSetting setting;
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

class SOFA_SOFAGUI_API FixOperation : public Operation
{
public:
    FixOperation():stiffness(10000.0) {};
    virtual ~FixOperation() {};
    virtual void start() ;
    virtual void execution() ;
    virtual void end() ;

    void setStiffness(double s) {stiffness = s;}
    virtual double getStiffness() const { return stiffness;}

    static std::string getDescription() {return "Fix Picked particle";}
protected:
    double stiffness;
};


class SOFA_SOFAGUI_API AddFrameOperation : public Operation
{
public:
    virtual void start() ;
    virtual void execution() {};
    virtual void end() {};

    static std::string getDescription() {return "Add a Frame to a Skinned model";}
};

class SOFA_SOFAGUI_API AddSutureOperation : public Operation
{
public:
    AddSutureOperation():stiffness(10.0), damping(1.0) {};
    virtual ~AddSutureOperation() {};
    virtual void start();
    virtual void execution() {};
    virtual void end() {};
    virtual void endOperation();

    void setStiffness(double f) { stiffness = f;}
    virtual double getStiffness() const {return stiffness;}
    void setDamping(double f) {damping = f;}
    virtual double getDamping() const {return damping;}

    static std::string getDescription() {return "Add a spring to suture two points.";}
protected:
    double stiffness;
    double damping;
};

}
}

#endif
