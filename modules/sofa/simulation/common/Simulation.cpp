/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/simulation/common/FindByTypeVisitor.h>
#include <sofa/simulation/common/ExportGnuplotVisitor.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/InstrumentVisitor.h>
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
#include <sofa/simulation/common/xml/NodeElement.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/PipeProcess.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/gl/DrawManagerGL.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>


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


using namespace sofa::defaulttype;
Simulation::Simulation()
    : numMechSteps( initData(&numMechSteps,(unsigned) 1,"numMechSteps","Number of mechanical steps within one update step. If the update time step is dt, the mechanical time step is dt/numMechSteps.") ),
      nbSteps( initData(&nbSteps, (unsigned)0, "nbSteps", "Number of animation steps completed", true, false)),
      nbMechSteps( initData(&nbMechSteps, (unsigned)0, "nbMechSteps", "Number of mechanical steps completed", true, false)),
      gnuplotDirectory( initData(&gnuplotDirectory,std::string(""),"gnuplotDirectory","Directory where the gnuplot files will be saved")),
      instrumentInUse( initData( &instrumentInUse, -1, "instrumentinuse", "Numero of the instrument currently used")),
      paused(false),mDrawUtility(0l)
{
    /// By default, initialise the Viewer with openGL
    /// We can switch Viewer between GL, OGRE and OSG with the setDrawUtility method
    this->setDrawUtility(new sofa::helper::gl::DrawManagerGL() );
}


Simulation::~Simulation()
{
}
/// The (unique) simulation which controls the scene
std::auto_ptr<Simulation> Simulation::theSimulation;


sofa::helper::gl::DrawManager& Simulation::DrawUtility()
{
    return *mDrawUtility;
}


void Simulation::setDrawUtility(sofa::helper::gl::DrawManager * _NewDrawUtility)
{
    sofa::helper::gl::DrawManager * old = mDrawUtility;
    mDrawUtility = _NewDrawUtility;
    if(mDrawUtility != NULL)
        delete old;
}


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
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    root->execute<PrintVisitor>(params);
}

/// Print all object in the graph
void Simulation::exportXML ( Node* root, const char* fileName, bool compact )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    if ( fileName!=NULL )
    {
        std::ofstream out ( fileName );
        out << "<?xml version=\"1.0\"?>\n";

        XMLPrintVisitor print ( params /* PARAMS FIRST */, out,compact );
        root->execute ( print );
    }
    else
    {
        XMLPrintVisitor print ( params /* PARAMS FIRST */, std::cout,compact );
        root->execute ( print );
    }
}

/// Initialize the scene.
void Simulation::init ( Node* root )
{
    //cerr<<"Simulation::init"<<endl;
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();

    setContext( root->getContext());

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

    //Get the list of instruments present in the scene graph
    getInstruments(root);

    // propagate the visualization settings (showVisualModels, etc.) in the whole graph
    updateVisualContext(root,Node::ALLFLAGS);
    updateVisualContext(getVisualRoot(),Node::ALLFLAGS);
}


void Simulation::initNode( Node* node)
{
    if(!node)
    {
        return;
    }
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    assert( getSimulation()->getContext() != NULL );
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
    getInstruments(node);
}

void Simulation::getInstruments( Node *node)
{
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    InstrumentVisitor fetchInstrument(params);
    fetchInstrument.execute(node);
    instruments = fetchInstrument.getInstruments();
}

/// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
void Simulation::animate ( Node* root, double dt )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode(std::string("Step"));
#endif

    sofa::helper::AdvancedTimer::begin("Animate");

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params /* PARAMS FIRST */, &ev );
        root->execute ( act );
    }

    //std::cout << "animate\n";
    double startTime = root->getTime();
    double mechanicalDt = dt/numMechSteps.getValue();
    //double nextTime = root->getTime() + root->getDt();

    // CHANGE to support MasterSolvers : CollisionVisitor is now activated within AnimateVisitor
    //root->execute<CollisionVisitor>(params);

    AnimateVisitor act(params);
    act.setDt ( mechanicalDt );
    BehaviorUpdatePositionVisitor beh(params /* PARAMS FIRST */, root->getDt());
    for( unsigned i=0; i<numMechSteps.getValue(); i++ )
    {
        root->execute ( beh );
        root->execute ( act );
        root->setTime ( startTime + (i+1)* act.getDt() );
        getVisualRoot()->setTime ( root->getTime() );
        root->execute<UpdateSimulationContextVisitor>(params);  // propagate time
        getVisualRoot()->execute<UpdateSimulationContextVisitor>(params);
        nbMechSteps.setValue(nbMechSteps.getValue() + 1);
    }

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params /* PARAMS FIRST */, &ev );
        root->execute ( act );
    }

    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    root->execute<UpdateMappingVisitor>(params);
    sofa::helper::AdvancedTimer::step("UpdateMappingEndEvent");
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params /* PARAMS FIRST */, &ev );
        root->execute ( act );
    }
    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");

    sofa::helper::AdvancedTimer::stepBegin("UpdateBBox");
    root->execute<UpdateBoundingBoxVisitor>(params);
    sofa::helper::AdvancedTimer::stepEnd("UpdateBBox");
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode(std::string("Step"));
#endif
    nbSteps.setValue(nbSteps.getValue() + 1);

    sofa::helper::AdvancedTimer::end("Animate");
}


void Simulation::updateVisual ( Node* root, double dt )
{
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode(std::string("UpdateVisual"));
#endif
    sofa::helper::AdvancedTimer::begin("UpdateVisual");
    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    root->execute<UpdateMappingVisitor>(params);
    sofa::helper::AdvancedTimer::step("UpdateMappingEndEvent");
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params /* PARAMS FIRST */, &ev );
        root->execute ( act );
    }
    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");
    root->execute<VisualUpdateVisitor>(params);
    sofa::helper::AdvancedTimer::end("UpdateVisual");
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode(std::string("UpdateVisual"));
#endif
}

/// Reset to initial state
void Simulation::reset ( Node* root )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();

    root->execute<CleanupVisitor>(params);
    root->execute<ResetVisitor>(params);
    sofa::core::MechanicalParams mparams(*params);
    root->execute<MechanicalPropagatePositionAndVelocityVisitor>(&mparams);
    root->execute<UpdateMappingVisitor>(params);
    root->execute<VisualUpdateVisitor>(params);

    nbSteps.setValue(0);
    nbMechSteps.setValue(0);
}

/// Initialize the textures
void Simulation::initTextures ( Node* root )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    root->execute<VisualInitVisitor>(params);
    // Do a visual update now as it is not done in load() anymore
    /// \todo Separate this into another method?
    root->execute<VisualUpdateVisitor>(params);
}


/// Compute the bounding box of the scene.
void Simulation::computeBBox ( Node* root, SReal* minBBox, SReal* maxBBox, bool init )
{
    sofa::core::visual::VisualParams* vparams = sofa::core::visual::VisualParams::defaultInstance();
    VisualComputeBBoxVisitor act(vparams);
    if ( root )
        root->execute ( act );
    if (init)
    {
        minBBox[0] = (SReal)(act.minBBox[0]);
        minBBox[1] = (SReal)(act.minBBox[1]);
        minBBox[2] = (SReal)(act.minBBox[2]);
        maxBBox[0] = (SReal)(act.maxBBox[0]);
        maxBBox[1] = (SReal)(act.maxBBox[1]);
        maxBBox[2] = (SReal)(act.maxBBox[2]);
    }
    else
    {
        if ((SReal)(act.minBBox[0]) < minBBox[0] ) minBBox[0] = (SReal)(act.minBBox[0]);
        if ((SReal)(act.minBBox[1]) < minBBox[1] ) minBBox[1] = (SReal)(act.minBBox[1]);
        if ((SReal)(act.minBBox[2]) < minBBox[2] ) minBBox[2] = (SReal)(act.minBBox[2]);
        if ((SReal)(act.maxBBox[0]) > maxBBox[0] ) maxBBox[0] = (SReal)(act.maxBBox[0]);
        if ((SReal)(act.maxBBox[1]) > maxBBox[1] ) maxBBox[1] = (SReal)(act.maxBBox[1]);
        if ((SReal)(act.maxBBox[2]) > maxBBox[2] ) maxBBox[2] = (SReal)(act.maxBBox[2]);
    }
}

void Simulation::setPaused(bool paused)
{
    this->paused = paused;
}

bool Simulation::getPaused()
{
    return paused;
}

/// Update contexts. Required before drawing the scene if root flags are modified.
void Simulation::updateContext ( Node* root )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    root->execute<UpdateContextVisitor>(params);
}

/// Update only Visual contexts. Required before drawing the scene if root flags are modified.( can filter by specifying a specific element)
void Simulation::updateVisualContext ( Node* root, Node::VISUAL_FLAG FILTER)
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    UpdateVisualContextVisitor vis(params /* PARAMS FIRST */, FILTER);
    vis.execute(root);
}
/// Render the scene
void Simulation::draw ( Node* root, sofa::core::visual::VisualParams* vparams )
{
    if ( !root ) return;
    if (root->visualManager.empty())
    {
        vparams->pass() = sofa::core::visual::VisualParams::Std;
        VisualDrawVisitor act ( vparams );
        root->execute ( &act );
        vparams->pass() = sofa::core::visual::VisualParams::Transparent;
        VisualDrawVisitor act2 ( vparams );
        root->execute ( &act2 );
    }
    else
    {
        Node::Sequence<core::visual::VisualManager>::iterator begin = root->visualManager.begin(), end = root->visualManager.end(), it;
        for (it = begin; it != end; ++it)
            (*it)->preDrawScene(vparams);
        bool rendered = false; // true if a manager did the rendering
        for (it = begin; it != end; ++it)
            if ((*it)->drawScene(vparams))
            {
                rendered = true;
                break;
            }
        if (!rendered) // do the rendering
        {
            vparams->pass() = sofa::core::visual::VisualParams::Std;

            VisualDrawVisitor act ( vparams );
            root->execute ( &act );
            vparams->pass() = sofa::core::visual::VisualParams::Transparent;
            VisualDrawVisitor act2 ( vparams );
            root->execute ( &act2 );
        }
        Node::Sequence<core::visual::VisualManager>::reverse_iterator rbegin = root->visualManager.rbegin(), rend = root->visualManager.rend(), rit;
        for (rit = rbegin; rit != rend; ++rit)
            (*rit)->postDrawScene(vparams);
    }
}


/// Render the scene - shadow pass
void Simulation::drawShadows ( Node* root )
{
    if ( !root ) return;

    //VisualDrawVisitor act ( core::VisualModel::Shadow );
    //root->execute ( &act );
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
        ExportOBJVisitor act ( params /* PARAMS FIRST */, &fout );
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

        ExportOBJVisitor act ( params /* PARAMS FIRST */, &fout,&mtl );
        root->execute ( &act );
    }
}

void Simulation::dumpState ( Node* root, std::ofstream& out )
{
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    out<<root->getTime() <<" ";
    WriteStateVisitor ( params /* PARAMS FIRST */, out ).execute ( root );
    out<<endl;
}

/// Initialize gnuplot file output
void Simulation::initGnuplot ( Node* root )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    InitGnuplotVisitor v(params /* PARAMS FIRST */, gnuplotDirectory.getFullPath());
    root->execute( v );
}

/// Update gnuplot file output
void Simulation::exportGnuplot ( Node* root, double time )
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    ExportGnuplotVisitor expg ( params /* PARAMS FIRST */, time);
    root->execute ( expg );
}

/// Load a scene from a file
Node* Simulation::processXML(xml::BaseElement* xml, const char *filename)
{
    if ( xml==NULL )
    {
        return NULL;
    }
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();

    // We go the the current file's directory so that all relative path are correct
    helper::system::SetDirectory chdir ( filename );

#ifndef WIN32
    // Reset local settings to make sure that floating-point values are interpreted correctly
    setlocale(LC_ALL,"C");
    setlocale(LC_NUMERIC,"C");
#endif

    // 				std::cout << "Initializing objects"<<std::endl;
    sofa::simulation::xml::NodeElement* nodeElt = dynamic_cast<sofa::simulation::xml::NodeElement *>(xml);
    if( nodeElt==NULL||!(nodeElt->init()))
    {
        std::cerr << "Objects initialization failed."<<std::endl;
    }

    Node* root = dynamic_cast<Node*> ( xml->getObject() );
    if ( root == NULL )
    {
        std::cerr << "Objects initialization failed."<<std::endl;
        delete xml;
        return NULL;
    }

    // 				std::cout << "Initializing simulation "<<root->getName() <<std::endl;

    // Find the Simulation component in the scene
    FindByTypeVisitor<Simulation> findSimu(params);
    findSimu.execute(root);
    if( !findSimu.found.empty() )
        setSimulation( findSimu.found[0] );

    return root;
}

/// Load from a string in memory
Node* Simulation::loadFromMemory ( const char *filename, const char *data, unsigned int size )
{
    //::sofa::simulation::init();
    // 				std::cerr << "Loading simulation XML file "<<filename<<std::endl;
    xml::BaseElement* xml = xml::loadFromMemory (filename, data, size );

    Node* root = processXML(xml, filename);

    // 				std::cout << "load done."<<std::endl;
    delete xml;

    return root;
}


/// Load a scene from a file
Node* Simulation::loadFromFile ( const char *filename )
{
    //::sofa::simulation::init();
    // 				std::cerr << "Loading simulation XML file "<<filename<<std::endl;
    xml::BaseElement* xml = xml::loadFromFile ( filename );

    Node* root = processXML(xml, filename);

    // 				std::cout << "load done."<<std::endl;
    delete xml;

    return root;
}

/// Load a scene
Node* Simulation::load ( const char *filename )
{
    std::string ext = sofa::helper::system::SetDirectory::GetExtension(filename);
    if (ext == "php" || ext == "pscn")
    {
        std::string out="",error="";
        std::vector<std::string> args;


        //TODO : replace when PipeProcess will get file as stdin
        //at the moment, the filename is given as an argument
        args.push_back(std::string("-f" + std::string(filename)));
        //args.push_back("-w");
        std::string newFilename="";
        //std::string newFilename=filename;

        helper::system::FileRepository fp("PATH", ".");
#ifdef WIN32
        std::string command = "php.exe";
#else
        std::string command = "php";
#endif
        if (!fp.findFile(command,""))
        {
            std::cerr << "Simulation : Error : php not found in your PATH environment" << std::endl;
            return NULL;
        }

        sofa::helper::system::PipeProcess::executeProcess(command.c_str(), args,  newFilename, out, error);

        if(error != "")
        {
            std::cerr << "Simulation : load : "<< error << std::endl;
            if (out == "")
                return NULL;
        }

        return loadFromMemory(filename, out.c_str(), out.size());
    }

    if (ext == "scn" || ext == "xml")
    {
        return loadFromFile(filename);
    }

    std::cerr << "Simulation : Error : extension not handled" << std::endl;
    return NULL;

}

/// Delete a scene from memory. After this call the pointer is invalid
void Simulation::unload(Node * root)
{
    if ( !root ) return;
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    if (dynamic_cast<Node*>(this->getContext()) == root)
    {
        instruments.clear();
        instrumentInUse.setValue(-1);
        this->setContext(0);
    }
    root->detachFromGraph();
    root->execute<CleanupVisitor>(params);
    root->execute<DeleteVisitor>(params);
    //delete root; //We unload only, and don't destrory the Node
}
//      void Simulation::addStep ( )
//      {
//        nbSteps++;
//      }

} // namespace simulation

} // namespace sofa

