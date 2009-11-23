/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#ifndef SOFA_GUI_MOUSEOPERATIONS_H
#define SOFA_GUI_MOUSEOPERATIONS_H

#include "SofaGUI.h"
#include <iostream>
#include <vector>

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

    void setStiffness(double s) {stiffness = s;}
    virtual double getStiffness() const { return stiffness;}

    static std::string getDescription() {return "Attach an object to the Mouse";}
protected:
    double stiffness;
};

class SOFA_SOFAGUI_API InciseOperation : public Operation
{
public:
    InciseOperation():cpt (0) {};
    virtual ~InciseOperation() {};
    virtual void start() ;
    virtual void execution() ;
    virtual void end() ;
    virtual void endOperation() ;

    void setIncisionMethod (int m) {method = m;}
    void setSnapingBorderValue (int m) {snapingBorderValue = m;}
    void setSnapingValue (int m) {snapingValue = m;}

    virtual int getIncisionMethod() const { return method;}
    virtual int getSnapingBorderValue() const { return snapingBorderValue;}
    virtual int getSnapingValue() const { return snapingValue;}

    static std::string getDescription() {return "Incise along a path";}
protected:
    int method;
    int snapingBorderValue;
    int snapingValue;
    int cpt;
};

class SOFA_SOFAGUI_API RemoveOperation : public Operation
{
public:
    virtual ~RemoveOperation() {};
    virtual void start() ;
    virtual void execution() ;
    virtual void end() ;
    static std::string getDescription() {return "Remove a primitive";}
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

class SOFA_SOFAGUI_API InjectOperation : public Operation
{
public:
    InjectOperation():potentialValue(100.0) {};
    virtual ~InjectOperation() {};
    virtual void start();
    virtual void execution();
    virtual void end() ;

    void setPotentialValue(double f) {potentialValue = f;}
    virtual double getPotentialValue() const {; return potentialValue;}
    void setStateTag(std::string s) {stateTag = s;}
    virtual std::string getStateTag() const {; return stateTag;}

    static std::string getDescription() {return "Set action potential using the Mouse";}
protected:
    double potentialValue;
    std::string stateTag;
};
}
}

#endif
