/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/PrintVisitor.h>
#include <sofa/simulation/ExportGnuplotVisitor.h>
#include <sofa/simulation/InitVisitor.h>
#include <sofa/simulation/AnimateVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/ResetVisitor.h>
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/simulation/ExportVisualModelOBJVisitor.h>
#include <sofa/simulation/WriteStateVisitor.h>
#include <sofa/simulation/XMLPrintVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/CleanupVisitor.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/UpdateLinksVisitor.h>
#include <sofa/simulation/init.h>
#include <sofa/simulation/DefaultAnimationLoop.h>
#include <sofa/simulation/DefaultVisualManagerLoop.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/init.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/ComponentNameHelper.h>

#include <sofa/simulation/SceneLoaderFactory.h>


#include <sofa/simulation/events/SimulationInitStartEvent.h>
#include <sofa/simulation/events/SimulationInitDoneEvent.h>
#include <sofa/simulation/events/SimulationInitTexturesDoneEvent.h>


#include <fstream>
#include <cstring>

#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/system/FileSystem.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalProjectPositionAndVelocityVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalProjectPositionAndVelocityVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalPropagateOnlyPositionAndVelocityVisitor;

namespace sofa
{

namespace simulation
{

using namespace sofa::defaulttype;
using namespace sofa;

Simulation::Simulation()
{
    // Safety check; it could be elsewhere, but here is a good place, I guess.
    if (!sofa::simulation::core::isInitialized())
        sofa::helper::printUninitializedLibraryWarning("SofaSimulationCommon", "sofa::simulation::common::init()");

    name.setValue("Simulation");
}


Simulation::~Simulation()
{
}

/// The (unique) simulation which controls the scene
Simulation::SPtr Simulation::theSimulation;

void setSimulation ( Simulation* s )
{
    Simulation::theSimulation.reset(s);

}

Simulation* getSimulation()
{
    return Simulation::theSimulation.get();
}

/// Print all object in the graph
void Simulation::print ( Node* root )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
    root->execute<PrintVisitor>(params);
}

/// Print all object in the graph
void Simulation::exportXML ( Node* root, const char* fileName )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
    if ( fileName!=nullptr )
    {
        std::ofstream out ( fileName );
        out << "<?xml version=\"1.0\"?>\n";

        XMLPrintVisitor print ( params, out );
        root->execute ( print );
    }
    else
    {
        XMLPrintVisitor print ( params, std::cout );
        root->execute ( print );
    }
}

/// Print all object in the graph
void Simulation::exportGraph ( Node* root, const char* filename )
{
    if ( !root ) return;

    SceneLoader *exporter = SceneLoaderFactory::getInstance()->getExporterEntryFileName(filename);

    if (exporter)
    {
        exporter->write(root,filename);
    }
    else
    {
        // unable to write the file
        msg_error() << "exportGraph: extension ("<<sofa::helper::system::SetDirectory::GetExtension(filename)<<") not handled for export";
    }
}

/// Initialize the scene.
void Simulation::init ( Node* root )
{
    sofa::helper::AdvancedTimer::stepBegin("Simulation::init");
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();

    if (!root->getAnimationLoop())
    {
        msg_warning(root) << "An animation loop is required, but has not been found. Add an animation loop in the root "
                            "of your scene to fix this warning. The list of available animation loop components is: ["
        << sofa::core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::behavior::BaseAnimationLoop>()
        << "]. A component of type " << DefaultAnimationLoop::GetClass()->className << " will be automatically added for you.";
        
        DefaultAnimationLoop::SPtr aloop = sofa::core::objectmodel::New<DefaultAnimationLoop>(root);
        aloop->setName(root->getNameHelper().resolveName(aloop->getClassName(), sofa::core::ComponentNameHelper::Convention::python));
        root->addObject(aloop,sofa::core::objectmodel::TypeOfInsertion::AtBegin);
    }

    if(!root->getVisualLoop())
    {
        msg_info(root) << "A visual loop is required, but has not been found. Add a visual loop in the root "
                            "of your scene to fix this warning. The list of available visual loop components is: ["
        << sofa::core::ObjectFactory::getInstance()->listClassesDerivedFrom<sofa::core::visual::VisualLoop>()
        << "]. A component of type " << DefaultVisualManagerLoop::GetClass()->className << " will be automatically added for you.";

        DefaultVisualManagerLoop::SPtr vloop = sofa::core::objectmodel::New<DefaultVisualManagerLoop>(root);
        vloop->setName(root->getNameHelper().resolveName(vloop->getClassName(), sofa::core::ComponentNameHelper::Convention::python));
        root->addObject(vloop,sofa::core::objectmodel::TypeOfInsertion::AtBegin);
    }

    // all the objects have now been created, update the links
    root->execute<UpdateLinksVisitor>(params);

    initNode(root);

    root->execute<UpdateBoundingBoxVisitor>(params);

    // propagate the visualization settings (showVisualModels, etc.) in the whole graph
    updateVisualContext(root);
    sofa::helper::AdvancedTimer::stepEnd("Simulation::init");
}


void Simulation::initNode( Node* node)
{
    if(!node)
    {
        return;
    }
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();

    SimulationInitStartEvent beginInit;
    PropagateEventVisitor pb {params, &beginInit};
    node->execute(pb);

    // apply the init() and bwdInit() methods to all the components.
    // and put the VisualModels in a separate graph, rooted at getVisualRoot()
    node->execute<InitVisitor>(params);

    SimulationInitDoneEvent endInit;
    PropagateEventVisitor pe {params, &endInit};
    node->execute(pe);

    {
        sofa::core::MechanicalParams mparams(*params);
        node->execute<MechanicalProjectPositionAndVelocityVisitor>(&mparams);
        node->execute<MechanicalPropagateOnlyPositionAndVelocityVisitor>(&mparams);
    }

    node->execute<StoreResetStateVisitor>(params);
}

/// Execute one timestep. If do is 0, the dt parameter in the graph will be used
void Simulation::animate ( Node* root, SReal dt )
{
    sofa::helper::AdvancedTimer::stepBegin("Simulation::animate");

    if ( !root ) {
        msg_error() << "Simulation::animate, no root found";
        return;
    }
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();

    sofa::core::behavior::BaseAnimationLoop* aloop = root->getAnimationLoop();
    if(aloop)
    {
        aloop->step(params,dt);
    }
    else
    {
        msg_error() << "Simulation::animate: AnimationLoop expected at the root node";
        return;
    }

    sofa::helper::AdvancedTimer::stepEnd("Simulation::animate");
}

void Simulation::updateVisual ( Node* root)
{
    sofa::helper::AdvancedTimer::stepBegin("Simulation::updateVisual");

    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
    sofa::core::visual::VisualLoop* vloop = root->getVisualLoop();

    if(vloop)
    {
        vloop->updateStep(params);
    }
    else
    {
        msg_error() << "Simulation::updateVisual: VisualLoop expected at the root node";
        return;
    }

    sofa::helper::AdvancedTimer::stepEnd("Simulation::updateVisual");
}

/// Reset to initial state
void Simulation::reset ( Node* root )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();

    // start by resetting the time
    const sofa::core::behavior::BaseAnimationLoop *animLoop = root->getAnimationLoop();
    if (animLoop)
        root->setTime(animLoop->getResetTime());
    else
        root->setTime(0.);
    UpdateSimulationContextVisitor(sofa::core::execparams::defaultInstance()).execute(root);

    // by definition cleanup() MUST only be called right before destroying the object
    // if for some reason some components need to do something, it has to be done in reset or storeResetState
    root->execute<ResetVisitor>(params);
    sofa::core::MechanicalParams mparams(*params);
    root->execute<MechanicalProjectPositionAndVelocityVisitor>(&mparams);
    root->execute<MechanicalPropagateOnlyPositionAndVelocityVisitor>(&mparams);
    root->execute<UpdateMappingVisitor>(params);
    root->execute<VisualUpdateVisitor>(params);
}

/// Initialize the textures
void Simulation::initTextures ( Node* root )
{

    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
    sofa::core::visual::VisualLoop* vloop = root->getVisualLoop();

    if(vloop)
    {
        vloop->initStep(params);
    }
    else
    {
        msg_error() << "Simulation::initTextures() : VisualLoop expected at the root node";
        return;
    }

    SimulationInitTexturesDoneEvent endInit;
    PropagateEventVisitor pe {params, &endInit};
    root->execute(pe);
}


/// Compute the bounding box of the scene.
void Simulation::computeBBox ( Node* root, SReal* minBBox, SReal* maxBBox, bool init )
{
    if ( !root ) return;
    sofa::core::visual::VisualParams* vparams = sofa::core::visual::visualparams::defaultInstance();
    sofa::core::visual::VisualLoop* vloop = root->getVisualLoop();
    if(vloop)
    {
        vloop->computeBBoxStep(vparams, minBBox, maxBBox, init);
    }
    else
    {
        msg_error() << "Simulation::computeBBox() : VisualLoop expected at the root node";
        return;
    }
}

/// Compute the bounding box of the scene.
void Simulation::computeTotalBBox ( Node* root, SReal* minBBox, SReal* maxBBox )
{
    assert ( root!=nullptr );
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
    root->execute<UpdateBoundingBoxVisitor>( params );
    type::BoundingBox bb = root->f_bbox.getValue();
    for(int i=0; i<3; i++){
        minBBox[i]= bb.minBBox()[i];
        maxBBox[i]= bb.maxBBox()[i];
    }
}

/// Update contexts. Required before drawing the scene if root flags are modified.
void Simulation::updateContext ( Node* root )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
    root->execute<UpdateContextVisitor>(params);
}

/// Update only Visual contexts. Required before drawing the scene if root flags are modified.( can filter by specifying a specific element)
void Simulation::updateVisualContext (Node* root)
{
    if ( !root ) return;
    sofa::core::visual::VisualParams* vparams = sofa::core::visual::visualparams::defaultInstance();
    sofa::core::visual::VisualLoop* vloop = root->getVisualLoop();

    if(vloop)
    {
        vloop->updateContextStep(vparams);
    }
    else
    {
        msg_error() << "Simulation::updateVisualContext() : VisualLoop expected at the root node";
        return;
    }
}
/// Render the scene
void Simulation::draw ( sofa::core::visual::VisualParams* vparams, Node* root )
{
    sofa::helper::AdvancedTimer::stepBegin("Simulation::draw");

    for(auto& visualLoop : root->getTreeObjects<sofa::core::visual::VisualLoop>())
    {
        if (!vparams) vparams = sofa::core::visual::visualparams::defaultInstance();
        vparams->update();
        visualLoop->drawStep(vparams);
    }

    sofa::helper::AdvancedTimer::stepEnd("Simulation::draw");
}

/// Export a scene to an OBJ 3D Scene
void Simulation::exportOBJ ( Node* root, const char* filename, bool exportMTL )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
    std::ofstream fout ( filename );

    fout << "# Generated from SOFA Simulation" << std::endl;

    if ( !exportMTL )
    {
        ExportVisualModelOBJVisitor act ( params, &fout );
        root->execute ( &act );
    }
    else
    {
        const char *path1 = strrchr ( filename, '/' );
        const char *path2 = strrchr ( filename, '\\' );
        const char* path = ( path1==nullptr ) ? ( ( path2==nullptr ) ?filename : path2+1 ) : ( path2==nullptr ) ? path1+1 : ( ( path1-filename ) > ( path2-filename ) ) ? path1+1 : path2+1;

        const char *ext = strrchr ( path, '.' );

        if ( !ext ) ext = path + strlen ( path );
        std::string mtlfilename ( path, ext );
        mtlfilename += ".mtl";
        std::string mtlpathname ( filename, ext );
        mtlpathname += ".mtl";
        std::ofstream mtl ( mtlpathname.c_str() );
        mtl << "# Generated from SOFA Simulation" << std::endl;
        fout << "mtllib "<<mtlfilename<<'\n';

        ExportVisualModelOBJVisitor act ( params, &fout,&mtl );
        root->execute ( &act );
    }
}

void Simulation::dumpState ( Node* root, std::ofstream& out )
{
    sofa::helper::ScopedAdvancedTimer dumpStateTimer("dumpState");

    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
    out<<root->getTime() <<" ";
    WriteStateVisitor ( params, out ).execute ( root );
    out<<std::endl;
}



/// Load a scene from a file
Node::SPtr Simulation::load ( const std::string& filename, bool reload, const std::vector<std::string>& sceneArgs )
{
    if( sofa::helper::system::SetDirectory::GetFileName(filename.c_str()).empty())
    {
        msg_error() << "Cannot load file '" << filename << "': filename cannot be extracted from the given path";
        return nullptr;
    }

    if (sofa::helper::system::SetDirectory::GetExtension(filename.c_str()).empty() )
    {
        msg_error() << "Cannot load file '" << filename << "': extension cannot be extracted from the given path";
        return nullptr;
    }

    if (!sofa::helper::system::FileSystem::exists(filename))
    {
        msg_error() << "Cannot load file '" << filename << "': file cannot be found";
        return nullptr;
    }

    SceneLoader *loader = SceneLoaderFactory::getInstance()->getEntryFileName(filename);

    if (loader) return loader->load(filename, reload, sceneArgs);

    const std::string extension = sofa::helper::system::SetDirectory::GetExtension(filename.c_str());
    if (extension == "py" || extension == "py3"
        || extension == "pyscn" || extension == "py3scn") //special case for Python extensions
    {
        msg_error() << "Cannot load file '" << filename << "': extension (" << extension << ") is only supported if the"
            " plugin SofaPython3 is loaded. SofaPython3 must be loaded first before being able to load the file.";
    }
    else
    {
        msg_error() << "Cannot load file '" << filename << "': extension (" << extension << ") not supported";
    }
    return nullptr;
}

/// Delete a scene from memory. After this call the pointer is invalid
void Simulation::unload(Node::SPtr root)
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
    root->detachFromGraph();
    root->execute<CleanupVisitor>(params);
    root->execute<DeleteVisitor>(params);
}

} // namespace simulation

} // namespace sofa
