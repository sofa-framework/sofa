#ifndef SOFA_GUI_GUIMANAGER_H
#define SOFA_GUI_GUIMANAGER_H

#include <sofa/helper/system/config.h>
#include <sofa/simulation/common/Node.h>
#include <vector>
#include <string>
#include <list>

#ifdef SOFA_BUILD_GUIMANAGER
#	define SOFA_GUIMANAGER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_GUIMANAGER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif


namespace sofa
{

namespace gui
{
class SofaGUI;


class SOFA_GUIMANAGER_API GUIManager
{
public:
    typedef int InitGUIFn(const char* name, const std::vector<std::string>& options);
    typedef SofaGUI* CreateGUIFn(const char* name, const std::vector<std::string>& options, sofa::simulation::Node::SPtr groot, const char* filename);

    struct GUICreator
    {
        const char* name;
        InitGUIFn* init;
        CreateGUIFn* creator;
        int priority;
    };
    static int Init(const char* argv0, const char* name ="");

    /*!
     *  \brief Set parameter for a gui creation and Store in the guiCreators list
     *
     *  \param name :     It is the name of your gui. This name is compared with the name parameter when you set GUIManager::Init(name). It must be the same.
     *  \param creator :  The pointer function which call when GUIManager::createGUI()
     *  \param init :     The pointer function which call when GUIManager::Init()
     *  \param priority : If nothing is given as name GUIManager::Init parameter GUIManager::valid_guiname is automaticly set compared with the priority
     *  \return 1 if the name is already used (failed), 0 if restry succed
     */
    static int RegisterGUI(const char* name, CreateGUIFn* creator, InitGUIFn* init=NULL, int priority=0);
    static const char* GetValidGUIName();
    static const std::string& GetCurrentGUIName();
    static std::vector<std::string> ListSupportedGUI();
    static std::string ListSupportedGUI(char separator);
    static void AddGUIOption(const char* option);
    static int createGUI(sofa::simulation::Node::SPtr groot = NULL, const char* filename = NULL);

    /// @name Static methods for direct access to GUI
    /// @{
    static int MainLoop(sofa::simulation::Node::SPtr groot = NULL, const char* filename = NULL);

    static void Redraw();

    static sofa::simulation::Node* CurrentSimulation();

    static void SetScene(sofa::simulation::Node::SPtr groot, const char* filename=NULL, bool temporaryFile=false);
    static void SetDimension(int  width , int  height );
    static void SetFullScreen();

    /// @}
protected:
    /*!
     *  \brief Comparaison between guiname passed as parameter and all guiname store in guiCreators list
     *  \param name : It is the name of your gui.
     *  \return NULL if the name don't match with any guiCreators name, the correct pointer otherwise
     */
    static GUICreator* GetGUICreator(const char* name = NULL);
    /* CLASS FIELDS */

    static std::list<GUICreator> guiCreators;

    static std::vector<std::string> guiOptions;
    static SofaGUI* currentGUI;
    static const char* valid_guiname;

public:
    static SofaGUI* getGUI();
};

}
}
#endif
