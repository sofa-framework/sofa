#include "fakegui.h"
#include <sofa/gui/GUIManager.h>


static sofa::gui::SofaGUI* CreateFakeGUI(const char* name, const std::vector<std::string>& options, sofa::simulation::Node::SPtr groot, const char* filename)
{
    return new FakeGUI();
}

void FakeGUI::Create()
{
    // sofa FakeGUI
    sofa::gui::GUIManager::RegisterGUI("fake",CreateFakeGUI);
    sofa::gui::GUIManager::Init(0,"fake");
    sofa::gui::GUIManager::createGUI(NULL,NULL);
}

void FakeGUI::sendMessage(const std::string & msgType,const std::string & msgValue)
{
    printf("FakeGUI::sendMessage(\"%s\",\"%s\")\n",msgType.c_str(),msgValue.c_str());
}


