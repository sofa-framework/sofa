/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/PrintVisitor.h>
#include <sofa/simulation/common/ExportGnuplotVisitor.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/simulation/common/ResetVisitor.h>
#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/simulation/common/ExportOBJVisitor.h>
#include <sofa/simulation/common/WriteStateVisitor.h>
#include <sofa/simulation/common/XMLPrintVisitor.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/UpdateMappingEndEvent.h>
#include <sofa/simulation/common/CleanupVisitor.h>
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/simulation/common/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/common/UpdateLinksVisitor.h>
#include <sofa/simulation/common/init.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/init.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/common/SceneLoaderFactory.h>


#include <fstream>
#include <string.h>




// #include <sofa/simulation/common/FindByTypeVisitor.h>



// #include <sofa/helper/system/FileRepository.h>

// #include <fstream>
// #include <string.h>
// #ifndef WIN32
// #include <locale.h>
// #endif



namespace sofa
{

namespace simulation
{

Node::SPtr Simulation::sRoot = NULL;

using namespace sofa::defaulttype;
Simulation::Simulation()
{
    // Safety check; it could be elsewhere, but here is a good place, I guess.
    if (!sofa::simulation::common::isInitialized())
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

sofa::simulation::Node::SPtr Simulation::GetRoot()
{
    return sRoot;
}

/// Print all object in the graph
void Simulation::print ( Node* root )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    root->execute<PrintVisitor>(params);
}

/// Print all object in the graph
void Simulation::exportXML ( Node* root, const char* fileName )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    if ( fileName!=NULL )
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
        serr << "exportGraph: extension ("<<sofa::helper::system::SetDirectory::GetExtension(filename)<<") not handled for export" << sendl;
    }
}


/// Initialize the scene.
void Simulation::init ( Node* root )
{
    //cerr<<"Simulation::init"<<endl;
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();

    //setContext( root->getContext());

    if (!root->getAnimationLoop())
    {
        root->getContext()->sout
                <<"Default Animation Manager Loop will be used. Add DefaultAnimationLoop to the root node of scene file to remove this warning"
                        <<root->getContext()->sendl;

        DefaultAnimationLoop::SPtr aloop = sofa::core::objectmodel::New<DefaultAnimationLoop>(root);
        aloop->setName(core::objectmodel::BaseObject::shortName(aloop.get()));
        root->addObject(aloop);
    }

    if(!root->getVisualLoop())
    {
        root->getContext()->sout
                <<"Default Visual Manager Loop will be used. Add DefaultVisualManagerLoop to the root node of scene file to remove this warning"
                        <<root->getContext()->sendl;

        DefaultVisualManagerLoop::SPtr vloop = sofa::core::objectmodel::New<DefaultVisualManagerLoop>(root);
        vloop->setName(core::objectmodel::BaseObject::shortName(vloop.get()));
        root->addObject(vloop);
    }

    // all the objects have now been created, update the links
    root->execute<UpdateLinksVisitor>(params);

    // apply the init() and bwdInit() methods to all the components.
    // and put the VisualModels in a separate graph, rooted at getVisualRoot()
    root->execute<InitVisitor>(params);

    // Save reset state for later uses in reset()
    root->execute<StoreResetStateVisitor>(params);
    {
        // Why do we need  a copy of the params here ?
        sofa::core::MechanicalParams mparams(*params);
        root->execute<MechanicalPropagatePositionAndVelocityVisitor>(&mparams);
    }

    root->execute<UpdateBoundingBoxVisitor>(params);

    // propagate the visualization settings (showVisualModels, etc.) in the whole graph
    updateVisualContext(root);
}


void Simulation::initNode( Node* node)
{
    if(!node)
    {
        return;
    }
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    node->execute<InitVisitor>(params);

    //node->execute<MechanicalPropagatePositionAndVelocityVisitor>(params);
    //node->execute<MechanicalPropagateFreePositionVisitor>(params);
    {
        sofa::core::MechanicalParams mparams(*params);
        node->execute<MechanicalPropagatePositionAndVelocityVisitor>(&mparams);
        /*sofa::core::MultiVecCoordId xfree = sofa::core::VecCoordId::freePosition();
          mparams.x() = xfree;
          MechanicalPropagatePositionVisitor act(&mparams   // PARAMS FIRST //, 0, xfree, true);
          node->execute(act);*/
    }

    node->execute<StoreResetStateVisitor>(params);
}

/// Execute one timestep. If do is 0, the dt parameter in the graph will be used
void Simulation::animate ( Node* root, SReal dt )
{
    sofa::helper::AdvancedTimer::stepBegin("Simulation::animate");

    if ( !root ) {
        serr<<"Simulation::animate, no root found"<<sendl;
        return;
    }
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();

    sofa::core::behavior::BaseAnimationLoop* aloop = root->getAnimationLoop();
    if(aloop)
    {
        aloop->step(params,dt);
    }
    else
    {
        serr<<"ERROR in Simulation::animate(): AnimationLoop expected at the root node"<<sendl;
        return;
    }

    sofa::helper::AdvancedTimer::stepEnd("Simulation::animate");
}

void Simulation::updateVisual ( Node* root)
{
    sofa::helper::AdvancedTimer::stepBegin("Simulation::updateVisual");

    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    core::visual::VisualLoop* vloop = root->getVisualLoop();

    if(vloop)
    {
        vloop->updateStep(params);
    }
    else
    {
        serr<<"ERROR in updateVisual(): VisualLoop expected at the root node"<<sendl;
        return;
    }

    sofa::helper::AdvancedTimer::stepEnd("Simulation::updateVisual");
}

/// Reset to initial state
void Simulation::reset ( Node* root )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();

    // start by resetting the time
    const core::behavior::BaseAnimationLoop *animLoop = root->getAnimationLoop();
    if (animLoop)
        root->setTime(animLoop->getResetTime());
    else
        root->setTime(0.);
    UpdateSimulationContextVisitor(sofa::core::ExecParams::defaultInstance()).execute(root);

//    root->execute<CleanupVisitor>(params);
    // by definition cleanup() MUST only be called right before destroying the object
    // if for some reason some components need to do something, it has to be done in reset or storeResetState

    root->execute<ResetVisitor>(params);
    sofa::core::MechanicalParams mparams(*params);
    root->execute<MechanicalPropagatePositionAndVelocityVisitor>(&mparams);
    root->execute<UpdateMappingVisitor>(params);
    root->execute<VisualUpdateVisitor>(params);
}

/// Initialize the textures
void Simulation::initTextures ( Node* root )
{

    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    core::visual::VisualLoop* vloop = root->getVisualLoop();

    if(vloop)
    {
        vloop->initStep(params);
    }
    else
    {
        serr<<"ERROR in initTextures() : VisualLoop expected at the root node"<<sendl;
        return;
    }
}


/// Compute the bounding box of the scene.
void Simulation::computeBBox ( Node* root, SReal* minBBox, SReal* maxBBox, bool init )
{
	if ( !root ) return;
    sofa::core::visual::VisualParams* vparams = sofa::core::visual::VisualParams::defaultInstance();
    core::visual::VisualLoop* vloop = root->getVisualLoop();
    if(vloop)
    {
        vloop->computeBBoxStep(vparams, minBBox, maxBBox, init);
    }
    else
    {
        serr<<"ERROR in computeBBox() : VisualLoop expected at the root node"<<sendl;
        return;
    }
}

/// Compute the bounding box of the scene.
void Simulation::computeTotalBBox ( Node* root, SReal* minBBox, SReal* maxBBox )
{
    assert ( root!=NULL );
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    root->execute<UpdateBoundingBoxVisitor>( params );
    defaulttype::BoundingBox bb = root->f_bbox.getValue();
    for(int i=0; i<3; i++){
        minBBox[i]= bb.minBBox()[i];
        maxBBox[i]= bb.maxBBox()[i];
    }
}

/// Update contexts. Required before drawing the scene if root flags are modified.
void Simulation::updateContext ( Node* root )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    root->execute<UpdateContextVisitor>(params);
}

/// Update only Visual contexts. Required before drawing the scene if root flags are modified.( can filter by specifying a specific element)
void Simulation::updateVisualContext (Node* root)
{
    if ( !root ) return;
    sofa::core::visual::VisualParams* vparams = sofa::core::visual::VisualParams::defaultInstance();
    core::visual::VisualLoop* vloop = root->getVisualLoop();

    if(vloop)
    {
        vloop->updateContextStep(vparams);
    }
    else
    {
        serr<<"ERROR in updateVisualContext() : VisualLoop expected at the root node"<<sendl;
        return;
    }

    /*
    UpdateVisualContextVisitor vis(vparams);
    vis.execute(root);*/
}
/// Render the scene
void Simulation::draw ( sofa::core::visual::VisualParams* vparams, Node* root )
{
    sofa::helper::AdvancedTimer::begin("Animate");
    sofa::helper::AdvancedTimer::stepBegin("Simulation::draw");

    core::visual::VisualLoop* vloop = root->getVisualLoop();
    if(vloop)
    {
        if (!vparams) vparams = sofa::core::visual::VisualParams::defaultInstance();
        vparams->update();

        vloop->drawStep(vparams);
    }
    else
    {
        serr<<"ERROR in draw() : VisualLoop expected at the root node"<<sendl;
        return;
    }

    sofa::helper::AdvancedTimer::stepEnd("Simulation::draw");
    sofa::helper::AdvancedTimer::end("Animate");
}

/// Export a scene to an OBJ 3D Scene
void Simulation::exportOBJ ( Node* root, const char* filename, bool exportMTL )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    std::ofstream fout ( filename );

    fout << "# Generated from SOFA Simulation" << std::endl;

    if ( !exportMTL )
    {
        ExportOBJVisitor act ( params, &fout );
        root->execute ( &act );
    }
    else
    {
        const char *path1 = strrchr ( filename, '/' );
        const char *path2 = strrchr ( filename, '\\' );
        const char* path = ( path1==NULL ) ? ( ( path2==NULL ) ?filename : path2+1 ) : ( path2==NULL ) ? path1+1 : ( ( path1-filename ) > ( path2-filename ) ) ? path1+1 : path2+1;

        const char *ext = strrchr ( path, '.' );

        if ( !ext ) ext = path + strlen ( path );
        std::string mtlfilename ( path, ext );
        mtlfilename += ".mtl";
        std::string mtlpathname ( filename, ext );
        mtlpathname += ".mtl";
        std::ofstream mtl ( mtlpathname.c_str() );
        mtl << "# Generated from SOFA Simulation" << std::endl;
        fout << "mtllib "<<mtlfilename<<'\n';

        ExportOBJVisitor act ( params, &fout,&mtl );
        root->execute ( &act );
    }
}

void Simulation::dumpState ( Node* root, std::ofstream& out )
{
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    out<<root->getTime() <<" ";
    WriteStateVisitor ( params, out ).execute ( root );
    out<<endl;
}



/// Load a scene from a file
Node::SPtr Simulation::load ( const char *filename )
{
    if( sofa::helper::system::SetDirectory::GetFileName(filename).empty() || // no filename
            sofa::helper::system::SetDirectory::GetExtension(filename).empty() ) // filename with no extension
        return NULL;

    SceneLoader *loader = SceneLoaderFactory::getInstance()->getEntryFileName(filename);

    if (loader)
    {
        sRoot = loader->load(filename);
        return sRoot;
    }

    // unable to load file
    serr << "extension ("<<sofa::helper::system::SetDirectory::GetExtension(filename)<<") not handled" << sendl;
    return NULL;
}

/// Delete a scene from memory. After this call the pointer is invalid
void Simulation::unload(Node::SPtr root)
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    //if (this->getContext() == (BaseContext*)root)
    //{
    //    this->setContext(NULL);
    //}
    root->detachFromGraph();
    root->execute<CleanupVisitor>(params);
    root->execute<DeleteVisitor>(params);
}

} // namespace simulation

} // namespace sofa
