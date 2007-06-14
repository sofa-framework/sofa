#ifndef SOFA_GUI_SOFAGUI_H
#define SOFA_GUI_SOFAGUI_H

#include <sofa/simulation/tree/GNode.h>

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

    static int MainLoop(sofa::simulation::tree::GNode* groot = NULL, const char* filename = NULL);

    static SofaGUI* CurrentGUI();

    static void Redraw();

    static sofa::simulation::tree::GNode* CurrentSimulation();

    /// @}

public:

    /// @name methods each GUI must implement
    /// @{

    SofaGUI();

    virtual int mainLoop()=0;
    virtual void redraw()=0;
    virtual int closeGUI()=0;

    virtual sofa::simulation::tree::GNode* currentSimulation()=0;

    /// @}

    /// @name registration of each GUI
    /// @{

    typedef int InitGUIFn(const char* name, const std::vector<std::string>& options);
    typedef SofaGUI* CreateGUIFn(const char* name, const std::vector<std::string>& options, sofa::simulation::tree::GNode* groot, const char* filename);
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
