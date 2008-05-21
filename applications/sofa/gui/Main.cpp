#include "SofaGUI.h"

namespace sofa
{

namespace gui
{

SOFA_LINK_CLASS(BatchGUI)

#ifdef SOFA_GUI_GLUT
SOFA_LINK_CLASS(SimpleGUI)
#endif

#ifdef SOFA_GUI_QGLVIEWER
SOFA_LINK_CLASS(QGLViewerGUI)
#endif
#ifdef SOFA_GUI_QTVIEWER
SOFA_LINK_CLASS(QTGUI)
#endif
#ifdef SOFA_GUI_QTOGREVIEWER
SOFA_LINK_CLASS(OgreGUI)
#endif

int SofaGUI::Init()
{
    if (guiCreators().empty())
    {
        std::cerr << "ERROR(SofaGUI): No GUI registered."<<std::endl;
        return 1;
    }
    const char* name = GetGUIName();
    if (currentGUI)
        return 0; // already initialized

    GUICreator* creator = GetGUICreator(name);
    if (!creator)
    {
        return 1;
    }
    if (creator->init)
        return (*creator->init)(name, guiOptions);
    else
        return 0;
}

int SofaGUI::MainLoop(sofa::simulation::Node* groot, const char* filename)
{
    int ret = 0;
    const char* name = GetGUIName();
    if (!currentGUI)
    {
        GUICreator* creator = GetGUICreator(name);
        if (!creator)
        {
            return 1;
        }
        currentGUI = (*creator->creator)(name, guiOptions, groot, filename);
        if (!currentGUI)
        {
            std::cerr << "ERROR(SofaGUI): GUI "<<name<<" creation failed."<<std::endl;
            return 1;
        }
    }
    ret = currentGUI->mainLoop();
    if (ret)
    {
        std::cerr << "ERROR(SofaGUI): GUI "<<name<<" main loop failed (code "<<ret<<")."<<std::endl;
        return ret;
    }
    return ret;
}

} // namespace gui

} // namespace sofa
