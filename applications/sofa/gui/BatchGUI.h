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
#ifndef SOFA_GUI_BATCHGUI_H
#define SOFA_GUI_BATCHGUI_H

#include <sofa/gui/SofaGUI.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace gui
{

class SOFA_SOFAGUI_API BatchGUI : public SofaGUI
{

public:

    /// @name methods each GUI must implement
    /// @{

    BatchGUI();

    void setScene(sofa::simulation::Node::SPtr groot, const char* filename="", bool temporaryFile=false);

    int mainLoop();
    void redraw();
    int closeGUI();

    static void setNumIterations(unsigned int n) {nbIter=n;};
    sofa::simulation::Node* currentSimulation();

    /// @}

    /// @name registration of each GUI
    /// @{

    static int InitGUI(const char* name, const std::vector<std::string>& options);
    static SofaGUI* CreateGUI(const char* name, const std::vector<std::string>& options, sofa::simulation::Node::SPtr groot = NULL, const char* filename = NULL);

    static const unsigned int DEFAULT_NUMBER_OF_ITERATIONS;
    /// @}

protected:
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    ~BatchGUI();

    sofa::simulation::Node::SPtr groot;
    std::string filename;
    static unsigned int nbIter;
};

} // namespace gui

} // namespace sofa

#endif
