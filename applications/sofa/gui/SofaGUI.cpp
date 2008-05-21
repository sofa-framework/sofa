#include "SofaGUI.h"
#include <string.h>
namespace sofa
{

namespace gui
{

const char* SofaGUI::programName = NULL;
std::string SofaGUI::guiName = "";
std::vector<std::string> SofaGUI::guiOptions;
SofaGUI* SofaGUI::currentGUI = NULL;

//std::list<SofaGUI::GUICreator> SofaGUI::guiCreators;
std::list<SofaGUI::GUICreator>& SofaGUI::guiCreators()
{
    static std::list<SofaGUI::GUICreator> creators;
    return creators;
}

void SofaGUI::SetProgramName(const char* argv0)
{
    if (argv0)
        programName = argv0;
}

const char* SofaGUI::GetProgramName()
{
    return programName;
}

std::vector<std::string> SofaGUI::ListSupportedGUI()
{
    std::vector<std::string> names;
    for(std::list<GUICreator>::iterator it = guiCreators().begin(), itend = guiCreators().end(); it != itend; ++it)
    {
        names.push_back(it->name);
    }
    return names;
}

std::string SofaGUI::ListSupportedGUI(char separator)
{
    std::string names;
    bool first = true;
    for(std::list<GUICreator>::iterator it = guiCreators().begin(), itend = guiCreators().end(); it != itend; ++it)
    {
        if (!first) names += separator; else first = false;
        names += it->name;
    }
    return names;
}

const char* SofaGUI::GetGUIName()
{
    const char* name = guiName.c_str();
    if (!name[0] && !guiCreators().empty())
    {
        std::list<GUICreator>::iterator it = guiCreators().begin();
        std::list<GUICreator>::iterator itend = guiCreators().end();
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

void SofaGUI::SetGUIName(const char* name)
{
    guiName = name;
}

void SofaGUI::AddGUIOption(const char* option)
{
    guiOptions.push_back(option);
}

SofaGUI::GUICreator* SofaGUI::GetGUICreator(const char* name)
{
    if (!name) name = GetGUIName();
    std::list<GUICreator>::iterator it = guiCreators().begin();
    std::list<GUICreator>::iterator itend = guiCreators().end();
    while (it != itend && strcmp(name, it->name))
        ++it;
    if (it == itend)
    {
        std::cerr << "ERROR(SofaGUI): GUI "<<name<<" not found."<<std::endl;
        std::cerr << "Available GUIs:" << ListSupportedGUI(' ') << std::endl;
        return NULL;
    }
    else
        return &(*it);
}

SofaGUI* SofaGUI::CurrentGUI()
{
    return currentGUI;
}

void SofaGUI::Redraw()
{
    if (currentGUI) currentGUI->redraw();
}

sofa::simulation::Node* SofaGUI::CurrentSimulation()
{
    if (currentGUI)
        return currentGUI->currentSimulation();
    else
        return NULL;
}

int SofaGUI::RegisterGUI(const char* name, CreateGUIFn* creator, InitGUIFn* init, int priority)
{
    std::list<GUICreator>::iterator it = guiCreators().begin();
    std::list<GUICreator>::iterator itend = guiCreators().end();
    while (it != itend && strcmp(name, it->name))
        ++it;
    if (it != itend)
    {
        std::cerr << "ERROR(SofaGUI): GUI "<<name<<" duplicate registration."<<std::endl;
        return 1;
    }
    GUICreator entry;
    entry.name = name;
    entry.creator = creator;
    entry.init = init;
    entry.priority = priority;
    guiCreators().push_back(entry);
    return 0;
}

SofaGUI::SofaGUI()
{
    if (currentGUI)
        std::cerr << "WARNING(SofaGUI): multiple concurrent active gui." << std::endl;
    else
        currentGUI = this;
}

SofaGUI::~SofaGUI()
{
    if (currentGUI == this)
        currentGUI = NULL;
}

} // namespace gui

} // namespace sofa
