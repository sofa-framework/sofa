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
#ifndef SOFA_GUI_SOFAGUI_H
#define SOFA_GUI_SOFAGUI_H

#include <sofa/simulation/common/Node.h>

#include <list>
//class QWidget;

namespace sofa
{

namespace gui
{

class SofaGUI
{

public:

    /// @name Static methods for direct access to GUI
    /// @{

    static void SetProgramName(const char* argv0);

    static const char* GetProgramName();

    static std::vector<std::string> ListSupportedGUI();
    static std::string ListSupportedGUI(char separator);

    static const char* GetGUIName();

    static void SetGUIName(const char* name="");
    static void AddGUIOption(const char* option);

    static int Init();

    static int Init(const char* argv0, const char* name = "")
    {
        SetProgramName(argv0);
        SetGUIName(name);
        return Init();
    }

    static int createGUI(sofa::simulation::Node* groot = NULL, const char* filename = NULL);

    static int MainLoop(sofa::simulation::Node* groot = NULL, const char* filename = NULL);



    static SofaGUI* CurrentGUI();

    static void Redraw();

    static sofa::simulation::Node* CurrentSimulation();



    /// @}

public:

    /// @name methods each GUI must implement
    /// @{

    SofaGUI();

    virtual int mainLoop()=0;
    virtual void redraw()=0;
    virtual int closeGUI()=0;
    virtual void setScene(sofa::simulation::Node* groot, const char* filename=NULL, bool temporaryFile=false)=0;
    virtual void setDimension(int /* width */, int /* height */) {};
    virtual void setFullScreen() {};

    virtual sofa::simulation::Node* currentSimulation()=0;

    /// @}

    /// @name registration of each GUI
    /// @{

    typedef int InitGUIFn(const char* name, const std::vector<std::string>& options);
    typedef SofaGUI* CreateGUIFn(const char* name, const std::vector<std::string>& options, sofa::simulation::Node* groot, const char* filename);
    static int RegisterGUI(const char* name, CreateGUIFn* creator, InitGUIFn* init=NULL, int priority=0);

    /// @}

protected:
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    virtual ~SofaGUI();

    static const char* programName;
    static std::string guiName;
    static std::vector<std::string> guiOptions;
    static SofaGUI* currentGUI;

    struct GUICreator
    {
        const char* name;
        InitGUIFn* init;
        CreateGUIFn* creator;
        int priority;
    };
    //static std::list<GUICreator> guiCreators;
    static std::list<GUICreator>& guiCreators();

    static GUICreator* GetGUICreator(const char* name = NULL);
};

} // namespace gui

} // namespace sofa

#endif
