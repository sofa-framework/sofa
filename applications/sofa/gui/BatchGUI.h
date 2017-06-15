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
#ifndef SOFA_GUI_BATCHGUI_H
#define SOFA_GUI_BATCHGUI_H

#include <sofa/gui/BaseGUI.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace gui
{

class SOFA_SOFAGUI_API BatchGUI : public BaseGUI
{

public:

    /// @name methods each GUI must implement
    /// @{

    BatchGUI();

    void setScene(sofa::simulation::Node::SPtr groot, const char* filename="", bool temporaryFile=false);

    void resetScene();

    int mainLoop();
    void redraw();
    int closeGUI();

    static void setNumIterations(unsigned int n) {nbIter=n;}
    sofa::simulation::Node* currentSimulation();

    /// @}

    /// @name registration of each GUI
    /// @{

    static int InitGUI(const char* name, const std::vector<std::string>& options);
    static BaseGUI* CreateGUI(const char* name, const std::vector<std::string>& options, sofa::simulation::Node::SPtr groot = NULL, const char* filename = NULL);

    static const unsigned int DEFAULT_NUMBER_OF_ITERATIONS;
    /// @}

protected:
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    ~BatchGUI();

    void startDumpVisitor();
    void stopDumpVisitor();

    std::ostringstream m_dumpVisitorStream;

    sofa::simulation::Node::SPtr groot;
    std::string filename;
    static unsigned int nbIter;
};

} // namespace gui

} // namespace sofa

#endif
