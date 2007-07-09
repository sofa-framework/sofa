#include <iostream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/gui/SofaGUI.h>

#include <GL/glut.h>

#ifndef WIN32
#include <dlfcn.h>
bool loadPlugin(const char* filename)
{
    void *handle;
    handle=dlopen(filename, RTLD_LAZY);
    if (!handle)
    {
        std::cerr<<"Error loading plugin "<<filename<<": "<<dlerror()<<std::endl;
        return false;
    }
    std::cerr<<"Plugin "<<filename<<" loaded."<<std::endl;
    return true;
}
#else
bool loadPlugin(const char* filename)
{
    std::cerr << "Plugin loading not supported on this platform.\n";
    return false;
}
#endif

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    sofa::helper::BackTrace::autodump();


    glutInit(&argc,argv);

    sofa::gui::SofaGUI::SetProgramName(argv[0]);
    std::string fileName ;
    bool        startAnim = false;
    bool        printFactory = false;
    std::string gui = sofa::gui::SofaGUI::GetGUIName();
    std::vector<std::string> plugins;
    std::vector<std::string> files;

    std::string gui_help = "choose the UI (";
    gui_help += sofa::gui::SofaGUI::ListSupportedGUI('|');
    gui_help += ")";

    sofa::helper::parse(&files, "This is a SOFA application. Here are the command line arguments")
//	.option(&fileName,'f',"file","scene file")
    .option(&startAnim,'s',"start","start the animation loop")
    .option(&printFactory,'p',"factory","print factory logs")
    //.option(&nogui,'g',"nogui","use no gui, run a number of iterations and exit")
    .option(&gui,'g',"gui",gui_help.c_str())
    .option(&plugins,'l',"load","load given plugins")
    (argc,argv);

    if (!files.empty()) fileName = files[0];

    for (unsigned int i=0; i<plugins.size(); i++)
        loadPlugin(plugins[i].c_str());

    if (printFactory)
    {
        std::cout << "////////// FACTORY //////////" << std::endl;
        sofa::helper::printFactoryLog();
        std::cout << "//////// END FACTORY ////////" << std::endl;
    }

    if (int err=sofa::gui::SofaGUI::Init(argv[0],gui.c_str()))
        return err;

    sofa::simulation::tree::GNode* groot = NULL;

    if (fileName.empty())
    {
        fileName = "liver.scn";
        sofa::helper::system::DataRepository.findFile(fileName);
    }

    groot = sofa::simulation::tree::Simulation::load(fileName.c_str());

    if (groot==NULL)
    {
        groot = new sofa::simulation::tree::GNode;
        //return 1;
    }

    if (startAnim)
        groot->setAnimate(true);



    //=======================================
    // Run the main loop

    if (gui=="none")
    {
        if (groot==NULL)
        {
            std::cerr<<"Could not load file "<<fileName<<std::endl;
            return 1;
        }
        std::cout << "Computing 1000 iterations." << std::endl;
        for (int i=0; i<1000; i++)
        {
            sofa::simulation::tree::Simulation::animate(groot);
        }
        std::cout << "1000 iterations done." << std::endl;
    }
    else
    {
        if (int err=sofa::gui::SofaGUI::MainLoop(groot,fileName.c_str()))
            return err;
        groot = sofa::gui::SofaGUI::CurrentSimulation();
    }

    if (groot!=NULL)
        sofa::simulation::tree::Simulation::unload(groot);
    return 0;
}
