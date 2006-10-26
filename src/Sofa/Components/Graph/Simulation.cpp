#include "Simulation.h"
#include "../XML/XML.h"
#include "../Common/SetDirectory.h"
#include "../init.h"
#include "PrintAction.h"
#include "InitAction.h"
#include "AnimateAction.h"
#include "MechanicalAction.h"
#include "CollisionAction.h"
#include "UpdateContextAction.h"
#include "UpdateMappingAction.h"
#include "ResetAction.h"
#include "VisualAction.h"
#include "DeleteAction.h"
#include "ExportOBJAction.h"
#include "WriteStateAction.h"
#include "XMLPrintAction.h"

#include <fstream>

namespace Sofa
{

namespace Components
{

namespace Graph
{

using namespace Common;

/// Load a scene from a file
GNode* Simulation::load(const char *filename)
{
    ::Sofa::Components::init();
    std::cerr << "Loading simulation XML file "<<filename<<std::endl;
    XML::BaseNode* xml = XML::load(filename);
    if (xml==NULL)
    {
        return NULL;
    }

    // We go the the current file's directory so that all relative path are correct
    SetDirectory chdir(filename);

    std::cout << "Initializing objects"<<std::endl;
    if (!xml->init())
    {
        std::cerr << "Objects initialization failed."<<std::endl;
    }

    GNode* root = dynamic_cast<GNode*>(xml->getBaseObject());
    if (root == NULL)
    {
        std::cerr << "Objects initialization failed."<<std::endl;
        delete xml;
        return NULL;
    }

    std::cout << "Initializing simulation "<<root->getName()<<std::endl;

    //root->init();
    root->execute<InitAction>();

    // As mappings might be initialized after visual models, it is necessary to update them
    root->execute<VisualUpdateAction>();

    std::cout << "load done."<<std::endl;

    delete xml;

    return root;
}

/// Print all object in the graph
void Simulation::print(GNode* root)
{
    if (!root) return;
    root->execute<PrintAction>();
}

/// Print all object in the graph
void Simulation::printXML(GNode* root, const char* fileName)
{
    if (!root) return;
    if( fileName!=NULL )
    {
        std::ofstream out(fileName);
        XMLPrintAction print(out);
        root->execute(print);
    }
    else
    {
        XMLPrintAction print(std::cout);
        root->execute(print);
    }
}

/// Initialize the scene.
void Simulation::init(GNode* root)
{
    if (!root) return;
    root->execute<InitAction>();
}

/// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
void Simulation::animate(GNode* root, double dt)
{
    double nextTime = root->getTime() + root->getDt();
    if (!root) return;
    //std::cout << "animate\n";

    root->execute<CollisionAction>();

    AnimateAction act;
    act.setDt(dt);
    root->execute(act);

    root->execute<UpdateMappingAction>();
    root->execute<VisualUpdateAction>();
    root->setTime( nextTime );
}

/// Reset to initial state
void Simulation::reset(GNode* root)
{
    if (!root) return;
    root->execute<ResetAction>();
    root->execute<MechanicalPropagatePositionAndVelocityAction>();
    root->execute<UpdateMappingAction>();
    root->execute<VisualUpdateAction>();
}

/// Initialize the textures
void Simulation::initTextures(GNode* root)
{
    if (!root) return;
    root->execute<VisualInitTexturesAction>();
}


/// Compute the bounding box of the scene.
void Simulation::computeBBox(GNode* root, double* minBBox, double* maxBBox)
{
    VisualComputeBBoxAction act;
    if (root)
        root->execute(act);
    minBBox[0] = act.minBBox[0];
    minBBox[1] = act.minBBox[1];
    minBBox[2] = act.minBBox[2];
    maxBBox[0] = act.maxBBox[0];
    maxBBox[1] = act.maxBBox[1];
    maxBBox[2] = act.maxBBox[2];
}

/// Update contexts. Required before drawing the scene if root flags are modified.
void Simulation::updateContext(GNode* root)
{
    if (!root) return;
    root->execute<UpdateContextAction>();
}

/// Render the scene
void Simulation::draw(GNode* root)
{
    if (!root) return;
    //std::cout << "draw\n";
    root->execute<VisualDrawAction>();
}

/// Delete a scene from memory. After this call the pointer is invalid
void Simulation::unload(GNode* root)
{
    if (!root) return;
    root->execute<DeleteAction>();
    if (root->getParent()!=NULL)
        root->getParent()->removeChild(root);
    delete root;
}

/// Export a scene to an OBJ 3D Scene
void Simulation::exportOBJ(GNode* root, const char* filename, bool exportMTL)
{
    if (!root) return;
    std::ofstream fout(filename);

    fout << "# Generated from SOFA Simulation" << std::endl;

    if (!exportMTL)
    {
        ExportOBJAction act(&fout);
        root->execute(&act);
    }
    else
    {
        const char *path1 = strrchr(filename, '/');
        const char *path2 = strrchr(filename, '\\');
        const char* path = (path1==NULL) ? ((path2==NULL)?filename : path2+1) : (path2==NULL) ? path1+1 : ((path1-filename) > (path2-filename)) ? path1+1 : path2+1;

        const char *ext = strrchr(path, '.');

        if (!ext) ext = path + strlen(path);
        std::string mtlfilename(path, ext);
        mtlfilename += ".mtl";
        std::string mtlpathname(filename, ext);
        mtlpathname += ".mtl";
        std::ofstream mtl(mtlpathname.c_str());
        mtl << "# Generated from SOFA Simulation" << std::endl;
        fout << "mtllib "<<mtlfilename<<'\n';

        ExportOBJAction act(&fout,&mtl);
        root->execute(&act);
    }
}

void Simulation::dumpState( GNode* root, std::ofstream& out )
{
    out<<root->getTime()<<" ";
    WriteStateAction(out).execute(root);
    out<<endl;
}

} // namespace Graph

} // namespace Components

} // namespace Sofa

