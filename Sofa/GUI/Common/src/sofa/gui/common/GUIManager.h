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

#include <sofa/simulation/Node.h>
#include <sofa/simulation/fwd.h>

#include <vector>
#include <string>
#include <list>


namespace sofa::gui::common
{

class BaseGUI;
class ArgumentParser;

class SOFA_GUI_COMMON_API GUIManager
{
public:
    typedef BaseGUI* CreateGUIFn(const char* name, sofa::simulation::NodeSPtr groot, const char* filename);
    typedef int RegisterGUIParameters(ArgumentParser* argumentParser);

    struct GUICreator
    {
        const char* name;
        CreateGUIFn* creator;
        RegisterGUIParameters* parameters;
        int priority;
    };
    static int Init(const char* argv0, const char* name ="");

    /*!
     *  \brief Set parameter for a gui creation and Store in the guiCreators list
     *
     *  \param name :     It is the name of your gui. This name is compared with the name parameter when you set GUIManager::Init(name). It must be the same.
     *  \param creator :  The pointer function which call when GUIManager::createGUI()
     *  \param init :     The pointer function which call when GUIManager::Init()
     *  \param priority : If nothing is given as name GUIManager::Init parameter GUIManager::valid_guiname is automatically set compared with the priority
     *  \return 1 if the name is already used (failed), 0 if restry succeed
     */
    static int RegisterGUI(const char* name, CreateGUIFn* creator, RegisterGUIParameters* parameters=nullptr, int priority=0);
    static const char* GetValidGUIName();
    static const std::string& GetCurrentGUIName();
    static std::vector<std::string> ListSupportedGUI();
    static std::string ListSupportedGUI(char separator);
    static void RegisterParameters(ArgumentParser* parser);
    static int createGUI(sofa::simulation::NodeSPtr groot = nullptr, const char* filename = nullptr);
    static void closeGUI();

    /// @name Static methods for direct access to GUI
    /// @{
    static int MainLoop(sofa::simulation::NodeSPtr groot = nullptr, const char* filename = nullptr);

    static void Redraw();

    static sofa::simulation::Node* CurrentSimulation();

    static void SetScene(sofa::simulation::NodeSPtr groot, const char* filename=nullptr, bool temporaryFile=false);
    static void SetDimension(int  width , int  height );
    static void SetFullScreen();
    static void CenterWindow();
    static void SaveScreenshot(const char* filename);

    /// @}
protected:
    /*!
     *  \brief Comparison between guiname passed as parameter and all guiname store in guiCreators list
     *  \param name : It is the name of your gui.
     *  \return nullptr if the name don't match with any guiCreators name, the correct pointer otherwise
     */
    static GUICreator* GetGUICreator(const char* name = nullptr);
    /* CLASS FIELDS */

    static std::list<GUICreator> guiCreators;

    static std::vector<std::string> guiOptions;
    static BaseGUI* currentGUI;
    static std::string valid_guiname;
    static ArgumentParser* currentArgumentParser;
public:
    static BaseGUI* getGUI();
};

} // namespace sofa::gui::common
