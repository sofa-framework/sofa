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
#ifndef FAKEGUI_H
#define FAKEGUI_H

#include <sofa/gui/common/BaseGUI.h>
#include <sofa/simulation/Node.h>

/// this fake GUI is only meant to manage "sendMessage" from python scripts
class FakeGUI : public sofa::gui::common::BaseGUI
{
protected:
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    ~FakeGUI() override {}

public:
    /// @name methods each GUI must implement
    /// @{
    int mainLoop() override {return 0;}
    void redraw() override {}
    int closeGUI() override {return 0;}
    virtual void setScene(sofa::simulation::Node::SPtr /*groot*/, const char* /*filename*/=nullptr, bool /*temporaryFile*/=false) override {}
    sofa::simulation::Node* currentSimulation() override {return nullptr;}
    /// @}

    /// @name methods to communicate with the GUI
    /// @{
    virtual void sendMessage(const std::string & /*msgType*/,const std::string & /*msgValue*/) override;
    /// @}

    static void Create();

};





#endif // FAKEGUI_H
