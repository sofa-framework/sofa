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
#ifndef FAKEGUI_H
#define FAKEGUI_H

#include <sofa/gui/BaseGUI.h>

/// this fake GUI is only meant to manage "sendMessage" from python scripts
class FakeGUI : public sofa::gui::BaseGUI
{
protected:
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    ~FakeGUI() {}

public:
    /// @name methods each GUI must implement
    /// @{
    virtual int mainLoop() {return 0;}
    virtual void redraw() {}
    virtual int closeGUI() {return 0;}
    virtual void setScene(sofa::simulation::Node::SPtr /*groot*/, const char* /*filename*/=NULL, bool /*temporaryFile*/=false) {}
    virtual sofa::simulation::Node* currentSimulation() {return 0;}
    /// @}

    /// @name methods to communicate with the GUI
    /// @{
    virtual void sendMessage(const std::string & /*msgType*/,const std::string & /*msgValue*/);
    /// @}

    static void Create();

};





#endif // FAKEGUI_H
