/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/ArgumentParser.h>
#include <string>

using sofa::helper::ArgumentParser;

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

    static void setNumIterations(const std::string& nbIterInp) 
    {
        int inpLen= nbIterInp.length();
       
        if (nbIterInp == "infinite")
        {
            nbIter = -1;
        }
        else if (inpLen)
        {
            nbIter = std::stoi(nbIterInp);
        }
        else
        {
            nbIter = DEFAULT_NUMBER_OF_ITERATIONS;
        }
        
    }
    sofa::simulation::Node* currentSimulation();

    /// @}

    /// @name registration of each GUI
    /// @{

    static BaseGUI* CreateGUI(const char* name, sofa::simulation::Node::SPtr groot = NULL, const char* filename = NULL);
    static int RegisterGUIParameters(ArgumentParser* argumentParser);


    static const signed int DEFAULT_NUMBER_OF_ITERATIONS;
    /// @}

protected:
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    ~BatchGUI();

    void startDumpVisitor();
    void stopDumpVisitor();

    std::ostringstream m_dumpVisitorStream;

    sofa::simulation::Node::SPtr groot;
    std::string filename;
    static signed int nbIter;
    static std::string nbIterInp;
};

} // namespace gui

} // namespace sofa

#endif
