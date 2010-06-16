#ifndef SOFA_GUI_GUIMANAGER_H
#define SOFA_GUI_GUIMANAGER_H

#include <sofa/helper/system/config.h>
#include <vector>
#include <string>
#include <list>

#ifdef SOFA_BUILD_GUIMANAGER
#	define SOFA_GUIMANAGER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_GUIMANAGER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif


/* GUIManager is the only place where the different guis available for Sofa are registered
and thus links against all the gui libs.
Its original purpose is to allow dynamic linking of the gui by seperating the lib responsible
for the registration of guis and the one responsible for the definition of what a gui should do
in Sofa. Prior both these operations where done by sofagui.lib.
*/

namespace sofa
{

namespace simulation
{
class Node;
}

namespace gui
{
class SofaGUI;


class SOFA_GUIMANAGER_API GUIManager
{
public:
    typedef int InitGUIFn(const char* name, const std::vector<std::string>& options);
    typedef SofaGUI* CreateGUIFn(const char* name, const std::vector<std::string>& options, sofa::simulation::Node* groot, const char* filename);

    struct GUICreator
    {
        const char* name;
        InitGUIFn* init;
        CreateGUIFn* creator;
        int priority;
    };
    static int Init(const char* argv0, const char* name ="");
    static int RegisterGUI(const char* name, CreateGUIFn* creator, InitGUIFn* init=NULL, int priority=0);
    static const char* GetValidGUIName();
    static const std::string& GetCurrentGUIName();
    static std::vector<std::string> ListSupportedGUI();
    static std::string ListSupportedGUI(char separator);
    static void AddGUIOption(const char* option);
    static int createGUI(sofa::simulation::Node* groot = NULL, const char* filename = NULL);

    /// @name Static methods for direct access to GUI
    /// @{
    static int MainLoop(sofa::simulation::Node* groot = NULL, const char* filename = NULL);

    static void Redraw();

    static sofa::simulation::Node* CurrentSimulation();

    static void SetScene(sofa::simulation::Node* groot, const char* filename=NULL, bool temporaryFile=false);
    static void SetDimension(int  width , int  height );
    static void SetFullScreen();

    /// @}
protected:

    static GUICreator* GetGUICreator(const char* name = NULL);
    /* CLASS FIELDS */

    static std::list<GUICreator> guiCreators;

    static std::vector<std::string> guiOptions;
    static SofaGUI* currentGUI;
    static const char* valid_guiname;
};

}
}
#endif
