/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/simulation/tree/xml/XML.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/simulation/tree/init.h>
#include <sofa/simulation/tree/PrintAction.h>
#include <sofa/simulation/tree/ExportGnuplotAction.h>
#include <sofa/simulation/tree/InitAction.h>
#include <sofa/simulation/tree/AnimateAction.h>
#include <sofa/simulation/tree/MechanicalAction.h>
#include <sofa/simulation/tree/CollisionAction.h>
#include <sofa/simulation/tree/UpdateContextAction.h>
#include <sofa/simulation/tree/UpdateMappingAction.h>
#include <sofa/simulation/tree/ResetAction.h>
#include <sofa/simulation/tree/VisualAction.h>
#include <sofa/simulation/tree/DeleteAction.h>
#include <sofa/simulation/tree/ExportOBJAction.h>
#include <sofa/simulation/tree/WriteStateAction.h>
#include <sofa/simulation/tree/XMLPrintAction.h>
#include <sofa/simulation/tree/PropagateEventAction.h>
#include <sofa/simulation/tree/AnimateBeginEvent.h>
#include <sofa/simulation/tree/AnimateEndEvent.h>
#include <fstream>


namespace sofa
{

namespace simulation
{

namespace tree
{

using namespace sofa::defaulttype;

/// Load a scene from a file
GNode* Simulation::load(const char *filename)
{
    ::sofa::simulation::tree::init();
    std::cerr << "Loading simulation XML file "<<filename<<std::endl;
    xml::BaseElement* xml = xml::load(filename);
    if (xml==NULL)
    {
        return NULL;
    }

    // We go the the current file's directory so that all relative path are correct
    helper::system::SetDirectory chdir(filename);

    std::cout << "Initializing objects"<<std::endl;
    if (!xml->init())
    {
        std::cerr << "Objects initialization failed."<<std::endl;
    }

    GNode* root = dynamic_cast<GNode*>(xml->getObject());
    if (root == NULL)
    {
        std::cerr << "Objects initialization failed."<<std::endl;
        delete xml;
        return NULL;
    }

    std::cout << "Initializing simulation "<<root->getName()<<std::endl;

    //root->init();
    //exportXML( root, "toto.scn" );
    root->execute<InitAction>();

    // As mappings might be initialized after visual models, it is necessary to update them
    // BUGFIX (Jeremie A.): disabled as initTexture was not called yet, and the GUI might not even be up yet
    //root->execute<VisualUpdateAction>();

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
    if (!root) return;
    if (root->getMultiThreadSimulation())
        return;

    {
        AnimateBeginEvent ev(dt);
        PropagateEventAction act(&ev);
        root->execute(act);
    }

    //std::cout << "animate\n";
    double nextTime = root->getTime() + root->getDt();

    // CHANGE to support MasterSolvers : CollisionAction is now activated within AnimateAction
    //root->execute<CollisionAction>();

    AnimateAction act;
    act.setDt(dt);
    root->execute(act);
    root->setTime( nextTime );
    root->execute<UpdateContextAction>();

    root->execute<UpdateMappingAction>();
    root->execute<VisualUpdateAction>();

    {
        AnimateEndEvent ev(dt);
        PropagateEventAction act(&ev);
        root->execute(act);
    }
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
    // Do a visual update now as it is not done in load() anymore
    /// \todo Separate this into another method?
    root->execute<VisualUpdateAction>();
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
    VisualDrawAction act(VisualDrawAction::Std);
    root->execute(&act);
    VisualDrawAction act2(VisualDrawAction::Transparent);
    root->execute(&act2);
}

/// Render the scene - shadow pass
void Simulation::drawShadows(GNode* root)
{
    if (!root) return;
    //std::cout << "drawShadows\n";
    VisualDrawAction act(VisualDrawAction::Shadow);
    root->execute(&act);
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

/// Export a scene to XML
void Simulation::exportXML(GNode* root, const char* filename)
{
    if (!root) return;
    std::ofstream fout(filename);
    XMLPrintAction act(fout);
    root->execute(&act);
}

void Simulation::dumpState( GNode* root, std::ofstream& out )
{
    out<<root->getTime()<<" ";
    WriteStateAction(out).execute(root);
    out<<endl;
}

/// Initialize gnuplot file output
void Simulation::initGnuplot(GNode* root)
{
    if (!root) return;
    root->execute<InitGnuplotAction>();
}

/// Update gnuplot file output
void Simulation::exportGnuplot(GNode* root, double time)
{
    if (!root) return;
    ExportGnuplotAction expg(time);
    root->execute(expg);
}

} // namespace tree

} // namespace simulation

} // namespace sofa

