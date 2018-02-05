/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <fstream>

#include "GUIManager.h"
#include "BaseGUI.h"
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>
#include <SofaSimulationCommon/init.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/Utils.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/FileRepository.h>

using sofa::helper::system::FileSystem;
using sofa::helper::Utils;

namespace sofa
{

namespace gui
{

/*STATIC FIELD DEFINITIONS */
BaseGUI* GUIManager::currentGUI = NULL;
std::list<GUIManager::GUICreator> GUIManager::guiCreators;
const char* GUIManager::valid_guiname = NULL;
ArgumentParser* GUIManager::currentArgumentParser = NULL;


BaseGUI* GUIManager::getGUI()
{
    return currentGUI;
}

void GUIManager::RegisterParameters(ArgumentParser* argumentParser)
{
    currentArgumentParser = argumentParser;
    for(std::list<GUICreator>::iterator it =guiCreators.begin(), itend =guiCreators.end(); it != itend; ++it)
    {
        if (it->parameters)
            it->parameters(argumentParser);
    }
}

const std::string &GUIManager::GetCurrentGUIName()
{
    return currentGUI->GetGUIName();
}

int GUIManager::RegisterGUI(const char* name, CreateGUIFn* creator, RegisterGUIParameters* parameters, int priority)
{
    if(guiCreators.size())
    {
        std::list<GUICreator>::iterator it = guiCreators.begin();
        std::list<GUICreator>::iterator itend = guiCreators.end();
        while (it != itend && strcmp(name, it->name))
            ++it;
        if (it != itend)
        {
            msg_error("GUIManager") << "ERROR(GUIManager): GUI "<<name<<" duplicate registration.";
            return 1;
        }
    }

    GUICreator entry;
    entry.name = name;
    entry.creator = creator;
    entry.parameters = parameters;
    entry.priority = priority;
    guiCreators.push_back(entry);
    return 0;
}

std::vector<std::string> GUIManager::ListSupportedGUI()
{
    std::vector<std::string> names;
    for(std::list<GUICreator>::iterator it = guiCreators.begin(), itend = guiCreators.end(); it != itend; ++it)
    {
        names.push_back(it->name);
    }
    return names;
}

std::string GUIManager::ListSupportedGUI(char separator)
{
    std::string names;
    bool first = true;
    for(std::list<GUICreator>::iterator it =guiCreators.begin(), itend =guiCreators.end(); it != itend; ++it)
    {
        if (!first) names += separator; else first = false;
        names += it->name;
    }
    return names;
}

const char* GUIManager::GetValidGUIName()
{
    const char* name;
    std::string lastGuiFilename = BaseGUI::getConfigDirectoryPath() + "/lastUsedGUI.ini";
    if (guiCreators.empty())
    {

        msg_error("GUIManager") << "ERROR(SofaGUI): No GUI registered.";
        return NULL;
    }
    else
    {
        //Check the config file for the last used GUI type
        if(FileSystem::exists(lastGuiFilename))
        {
            std::string lastGuiName;
            std::ifstream lastGuiStream(lastGuiFilename.c_str());
            std::getline(lastGuiStream,lastGuiName);
            lastGuiStream.close();

            const char* lastGuiNameChar = lastGuiName.c_str();

            // const char* lastGuiNameChar = "qt";
            std::list<GUICreator>::iterator it1 = guiCreators.begin();
            std::list<GUICreator>::iterator itend1 = guiCreators.end();
            while(++it1 != itend1)
            {
                if( strcmp(lastGuiNameChar, it1->name) == 0 )
                {
                    return it1->name;
                }
            }
            msg_warning("GUIManager") << "WARNING(SofaGUI): Previously used GUI not registered. Using default GUI.";
        }
        else
        {
            msg_info("GUIManager") << "INFO(SofaGUI): lastUsedGUI.ini not found; using default GUI.";
        }

        std::list<GUICreator>::iterator it =guiCreators.begin();
        std::list<GUICreator>::iterator itend =guiCreators.end();
        name = it->name;
        int prio = it->priority;
        while (++it != itend)
        {
            if (it->priority > prio)
            {
                name = it->name;
                prio = it->priority;
            }
        }
    }
    return name;
}

GUIManager::GUICreator* GUIManager::GetGUICreator(const char* name)
{
    if (!name) name = GetValidGUIName();
    std::list<GUICreator>::iterator it =guiCreators.begin();
    std::list<GUICreator>::iterator itend =guiCreators.end();
    while (it != itend && strcmp(name, it->name))
        ++it;
    if (it == itend)
    {
        msg_error("GUIManager") << "GUI '"<<name<<"' creation failed."<< msgendl
                                << "Available GUIs: {" << ListSupportedGUI(' ') <<  "}";
        return NULL;
    }
    else
        return &(*it);
}

int GUIManager::Init(const char* argv0, const char* name)
{
    BaseGUI::SetProgramName(argv0);
    BaseGUI::SetArgumentParser(currentArgumentParser);
    sofa::simulation::common::init();

    static bool first = true;
    if (first)
    {
        sofa::component::initComponentBase();
        sofa::component::initComponentCommon();
        sofa::component::initComponentGeneral();
        sofa::component::initComponentAdvanced();
        sofa::component::initComponentMisc();

        first = false;
    }

    // Read the paths to the share/ and examples/ directories from etc/sofa.ini,
    const std::string etcDir = Utils::getSofaPathPrefix() + "/etc";
    const std::string sofaIniFilePath = etcDir + "/sofa.ini";
    std::map<std::string, std::string> iniFileValues = Utils::readBasicIniFile(sofaIniFilePath);

    // and add them to DataRepository
    if (iniFileValues.find("SHARE_DIR") != iniFileValues.end())
    {
        std::string shareDir = iniFileValues["SHARE_DIR"];
        if (!FileSystem::isAbsolute(shareDir))
            shareDir = etcDir + "/" + shareDir;
        sofa::helper::system::DataRepository.addFirstPath(shareDir);
    }
    if (iniFileValues.find("EXAMPLES_DIR") != iniFileValues.end())
    {
        std::string examplesDir = iniFileValues["EXAMPLES_DIR"];
        if (!FileSystem::isAbsolute(examplesDir))
            examplesDir = etcDir + "/" + examplesDir;
        sofa::helper::system::DataRepository.addFirstPath(examplesDir);
    }

    if (currentGUI)
        return 0; // already initialized

    if (guiCreators.empty())
    {
        msg_error("GUIManager") << "No GUI registered.";
        return 1;
    }

    if( strcmp(name,"") == 0 || name == NULL)
    {
        name = GetValidGUIName(); // get the default gui name
    }
    GUICreator *creator = GetGUICreator(name);
    if(!creator)
    {
        return 1;
    }
    valid_guiname = name; // at this point we must have a valid name for the gui.

    return 0;
}


int GUIManager::createGUI(sofa::simulation::Node::SPtr groot, const char* filename)
{
    if (!currentGUI)
    {
        GUICreator* creator = GetGUICreator(valid_guiname);
        if (!creator)
        {
            return 1;
        }
        currentGUI = (*creator->creator)(valid_guiname, groot, filename);
        if (!currentGUI)
        {
            msg_error("GUIManager") << "GUI '"<<valid_guiname<<"' creation failed." ;
            return 1;
        }
        //Save this GUI type as the last used GUI
        const std::string lastGuiFilePath = BaseGUI::getConfigDirectoryPath() + "/lastUsedGUI.ini";
        std::ofstream out(lastGuiFilePath.c_str(),std::ios::out);
        out << valid_guiname << std::endl;
        out.close();
    }
    return 0;
}

void GUIManager::closeGUI()
{
    if(currentGUI) currentGUI->closeGUI();
}

void GUIManager::Redraw()
{
    if (currentGUI) currentGUI->redraw();
}

sofa::simulation::Node* GUIManager::CurrentSimulation()
{
    if (currentGUI)
        return currentGUI->currentSimulation();
    else
        return NULL;
}

void GUIManager::SetScene(sofa::simulation::Node::SPtr groot, const char* filename /*=NULL*/, bool temporaryFile /*=false*/ )
{
    if (currentGUI)
    {
        currentGUI->setScene(groot,filename,temporaryFile);
        currentGUI->configureGUI(groot);
    }

}

int GUIManager::MainLoop(sofa::simulation::Node::SPtr groot, const char* filename)
{
    int ret = 0;
    if (!currentGUI)
    {
        createGUI(groot, filename);
    }
    ret = currentGUI->mainLoop();
    if (ret)
    {
        dmsg_error("GUIManager") << " GUI '"<<currentGUI->GetGUIName()<<"' main loop failed (code "<<ret<<").";
        return ret;
    }
    return ret;
}
void GUIManager::SetDimension(int  width , int  height )
{
    if (currentGUI)
    {
        //        std::string viewerFileName;
        //        std::string path = sofa::helper::system::DataRepository.getFirstPath();
        //        viewerFileName = path.append("/share/config/sofaviewer.ini");

        //        if(sofa::helper::system::DataRepository.findFile(viewerFileName))
        //        {
        //            std::string configPath = sofa::helper::system::DataRepository.getFile(viewerFileName);
        //            std::string w, h;
        //            std::ifstream viewerStream(configPath.c_str());
        //            std::getline(viewerStream,w);
        //            std::getline(viewerStream,h);
        //            viewerStream.close();

        //            std::stringstream convertW(w);
        //            convertW >> width;

        //            std::stringstream convertH(h);
        //            convertH >> height;
        //        }
        currentGUI->setViewerResolution(width,height);
    }
}
void GUIManager::SetFullScreen()
{
    if (currentGUI) currentGUI->setFullScreen();
    else{ msg_error("GUIManager") <<"no currentGUI" ; }
}
void GUIManager::SaveScreenshot(const char* filename)
{
    if (currentGUI) {
        std::string output = (filename?std::string(filename):"output.png");
        currentGUI->saveScreenshot(output);
    }
}


}
// namespace gui

}
// namespace sofa
