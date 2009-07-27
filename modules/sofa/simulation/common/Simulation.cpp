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
#include <sofa/simulation/common/CleanupVisitor.h>
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
#include <sofa/core/ObjectFactory.h>
#include <fstream>
#include <string.h>
namespace sofa
{

namespace simulation
{


using namespace sofa::defaulttype;
Simulation::Simulation()
    : numMechSteps( initData(&numMechSteps,(unsigned) 1,"numMechSteps","Number of mechanical steps within one update step. If the update time step is dt, the mechanical time step is dt/numMechSteps.") ),
      nbSteps(0),
      needToPrefetch(false),
      gnuplotDirectory( initData(&gnuplotDirectory,std::string(""),"gnuplotDirectory","Directory where the gnuplot files will be saved")),
      instrumentInUse( initData( &instrumentInUse, -1, "instrumentinuse", "Numero of the instrument currently used"))
{}

Simulation::~Simulation()
{
    if (Simulation::theSimulation == this) setSimulation( NULL );
}

/// The (unique) simulation which controls the scene
Simulation* Simulation::theSimulation = NULL;

void setSimulation ( Simulation* s )
{
    Simulation::theSimulation = s;
}

Simulation* getSimulation()
{
    return Simulation::theSimulation;
}

/// Print all object in the graph
void Simulation::print ( Node* root )
{
    if ( !root ) return;
    root->execute<PrintVisitor>();
}

/// Print all object in the graph
void Simulation::printXML ( Node* root, const char* fileName, bool compact )
{
    if ( !root ) return;
    if ( fileName!=NULL )
    {
        std::ofstream out ( fileName );
        XMLPrintVisitor print ( out,compact );
        root->execute ( print );
    }
    else
    {
        XMLPrintVisitor print ( std::cout,compact );
        root->execute ( print );
    }
}

/// Initialize the scene.
void Simulation::init ( Node* root )
{
    //cerr<<"Simulation::init"<<endl;
    setContext( root->getContext());
    if ( !root ) return;
    needToPrefetch = false;
    root->execute<InitVisitor>();
    // Save reset state for later uses in reset()
    root->execute<MechanicalPropagatePositionAndVelocityVisitor>();
    root->execute<MechanicalPropagateFreePositionVisitor>();
    root->execute<StoreResetStateVisitor>();

    //Get the list of instruments present in the scene graph
    getInstruments(root);

    nbSteps = 0;
}

void Simulation::getInstruments( Node *node)
{
    InstrumentVisitor fetchInstrument;
    fetchInstrument.execute(node);
    instruments = fetchInstrument.getInstruments();
}

/// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
void Simulation::animate ( Node* root, double dt )
{
    if ( !root ) return;
    if ( root->getMultiThreadSimulation() )
        return;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment(std::string("Begin Step"));
#endif
    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        root->execute ( act );
    }

    //std::cout << "animate\n";
    double startTime = root->getTime();
    double mechanicalDt = dt/numMechSteps.getValue();
    //double nextTime = root->getTime() + root->getDt();

    // CHANGE to support MasterSolvers : CollisionVisitor is now activated within AnimateVisitor
    //root->execute<CollisionVisitor>();

    AnimateVisitor act;
    act.setDt ( mechanicalDt );
    BehaviorUpdatePositionVisitor beh(root->getDt());
    for( unsigned i=0; i<numMechSteps.getValue(); i++ )
    {
        root->execute ( beh );
        root->execute ( act );
        root->setTime ( startTime + (i+1)* act.getDt() );
        root->execute<UpdateSimulationContextVisitor>();
    }

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        root->execute ( act );
    }

    root->execute<UpdateMappingVisitor>();
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        root->execute ( act );
    }
    root->execute<VisualUpdateVisitor>();
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printComment(std::string("End Step"));
#endif
    nbSteps++;
}


/// Reset to initial state
void Simulation::reset ( Node* root )
{
    if ( !root ) return;

    root->execute<ResetVisitor>();
    root->execute<MechanicalPropagatePositionAndVelocityVisitor>();
    root->execute<UpdateMappingVisitor>();
    root->execute<VisualUpdateVisitor>();

    nbSteps = 0;
}

/// Initialize the textures
void Simulation::initTextures ( Node* root )
{
    if ( !root ) return;
    root->execute<VisualInitVisitor>();
    // Do a visual update now as it is not done in load() anymore
    /// \todo Separate this into another method?
    root->execute<VisualUpdateVisitor>();
}


/// Compute the bounding box of the scene.
void Simulation::computeBBox ( Node* root, SReal* minBBox, SReal* maxBBox )
{
    VisualComputeBBoxVisitor act;
    if ( root )
        root->execute ( act );
    minBBox[0] = (SReal)(act.minBBox[0]);
    minBBox[1] = (SReal)(act.minBBox[1]);
    minBBox[2] = (SReal)(act.minBBox[2]);
    maxBBox[0] = (SReal)(act.maxBBox[0]);
    maxBBox[1] = (SReal)(act.maxBBox[1]);
    maxBBox[2] = (SReal)(act.maxBBox[2]);
}

/// Update contexts. Required before drawing the scene if root flags are modified.
void Simulation::updateContext ( Node* root )
{
    if ( !root ) return;
    root->execute<UpdateContextVisitor>();
}

/// Update only Visual contexts. Required before drawing the scene if root flags are modified.( can filter by specifying a specific element)
void Simulation::updateVisualContext ( Node* root, Node::VISUAL_FLAG FILTER)
{
    if ( !root ) return;
    UpdateVisualContextVisitor vis(FILTER);
    vis.execute(root);
}
/// Render the scene
void Simulation::draw ( Node* root, helper::gl::VisualParameters* params )
{
    if ( !root ) return;

    if (root->visualManager.empty())
    {
        VisualDrawVisitor act ( core::VisualModel::Std );
        root->execute ( &act );

        VisualDrawVisitor act2 ( core::VisualModel::Transparent );
        root->execute ( &act2 );
    }
    else
    {
        Node::Sequence<core::VisualManager>::iterator begin = root->visualManager.begin(), end = root->visualManager.end(), it;
        for (it = begin; it != end; ++it)
            (*it)->preDrawScene(params);
        bool rendered = false; // true if a manager did the rendering
        for (it = begin; it != end; ++it)
            if ((*it)->drawScene(params))
            {
                rendered = true;
                break;
            }
        if (!rendered) // do the rendering
        {
            VisualDrawVisitor act ( core::VisualModel::Std );
            root->execute ( &act );

            VisualDrawVisitor act2 ( core::VisualModel::Transparent );
            root->execute ( &act2 );
        }
        Node::Sequence<core::VisualManager>::reverse_iterator rbegin = root->visualManager.rbegin(), rend = root->visualManager.rend(), rit;
        for (rit = rbegin; rit != rend; ++rit)
            (*rit)->postDrawScene(params);
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
    std::ofstream fout ( filename );

    fout << "# Generated from SOFA Simulation" << std::endl;

    if ( !exportMTL )
    {
        ExportOBJVisitor act ( &fout );
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

        ExportOBJVisitor act ( &fout,&mtl );
        root->execute ( &act );
    }
}

/// Export a scene to XML
void Simulation::exportXML ( Node* root, const char* filename )
{
    if ( !root ) return;
    std::ofstream fout ( filename );
    XMLPrintVisitor act ( fout );
    root->execute ( &act );
}

void Simulation::dumpState ( Node* root, std::ofstream& out )
{
    out<<root->getTime() <<" ";
    WriteStateVisitor ( out ).execute ( root );
    out<<endl;
}

/// Initialize gnuplot file output
void Simulation::initGnuplot ( Node* root )
{
    if ( !root ) return;
    InitGnuplotVisitor v(gnuplotDirectory.getFullPath());
    root->execute( v );
}

/// Update gnuplot file output
void Simulation::exportGnuplot ( Node* root, double time )
{
    if ( !root ) return;
    ExportGnuplotVisitor expg ( time );
    root->execute ( expg );
}
void Simulation::unload(Node * root)
{
    if ( !root ) return;
    if (dynamic_cast<Node*>(this->getContext()) == root)
    {
        instruments.clear();
        instrumentInUse.setValue(-1);
    }
    root->execute<CleanupVisitor>();
    root->execute<DeleteVisitor>();
    root->detachFromGraph();
}
//      void Simulation::addStep ( )
//      {
//        nbSteps++;
//      }

} // namespace simulation

} // namespace sofa

