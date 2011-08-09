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
//
// C++ Implementation: GNodeVisitor
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/simulation/bgl/SMPBglSimulation.h>
#include <sofa/simulation/bgl/BglNode.h>

#include <sofa/core/ObjectFactory.h>

//#include <sofa/simulation/tree/SMPSimulation.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/common/PrintVisitor.h>
#include <sofa/simulation/common/FindByTypeVisitor.h>
#include <sofa/simulation/common/ExportGnuplotVisitor.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/simulation/common/ResetVisitor.h>
#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/simulation/common/ExportOBJVisitor.h>
#include <sofa/simulation/common/WriteStateVisitor.h>
#include <sofa/simulation/common/XMLPrintVisitor.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/UpdateMappingEndEvent.h>
#include <sofa/helper/system/PipeProcess.h>
#include <athapascan-1>
#include <Multigraph.inl>
#ifdef SOFA_SMP_WEIGHT
#include <Partitionner.h>
#endif
#include <fstream>
#include <string.h>

#include <sofa/helper/system/thread/CTime.h>
using
sofa::helper::system::thread::CTime;
using
sofa::helper::system::thread::ctime_t;

namespace sofa
{
namespace simulation
{
namespace bgl
{

Node *_root=NULL;
double _dt;
struct doCollideTask
{

    void operator()()
    {
        _root->execute<CollisionVisitor>();

    }
};
#ifdef SOFA_SMP_WEIGHT
struct compileGraphTask
{
    static Iterative::Multigraph<MainLoopTask> *mg;
    static  Iterative::Multigraph<MainLoopTask>*& getMultigraph()
    {
        static   Iterative::Multigraph<MainLoopTask> *mg=0;
        return mg;
    }
    static void setMultigraph(Iterative::Multigraph<MainLoopTask>* _mg)
    {
        Iterative::Multigraph<MainLoopTask> *&mg=getMultigraph();
        mg=_mg;
    }
    void operator()()
    {

        ctime_t  t0 = CTime::getRefTime ();
        if(getMultigraph()&&!getMultigraph()->compiled)
        {
            sofa::core::CallContext::ProcessorType etype=sofa::core::CallContext::getExecutionType();
            sofa::core::CallContext::setExecutionType(sofa::core::CallContext::GRAPH_KAAPI);
            getMultigraph()->compile();
            std::cerr<<"graph compiled"<<std::endl;
            sofa::core::CallContext::setExecutionType(etype);
            getMultigraph()->compiled=true;
            getMultigraph()->deployed=false;
        }
        ctime_t
        t1 = CTime::getRefTime ();
        std::cerr << "Compiling Time: " <<
                ((t1 - t0) / (CTime::getRefTicksPerSec () / 1000)) * 0.001<< std::endl;

    }
};
#endif
struct animateTask
{
    void operator()()
    {
        sofa::simulation::bgl::SMPBglSimulation* simulation=dynamic_cast<sofa::simulation::bgl::SMPBglSimulation *>(sofa::simulation::bgl::getSimulation());
        simulation->generateTasks( _root);
#ifdef SOFA_SMP_WEIGHT
        a1::Fork<compileGraphTask>(a1::SetSite(2))();
#endif

    }
};

struct collideTask
{
    void operator()()
    {

        a1::Fork<doCollideTask>()();

    }
};
struct visuTask
{
    void operator()()
    {
        //ColisonBeginEvent!!
        //_root->execute<CollisionVisitor>();
    }
};
struct MainLoopTask
{

    void operator()()
    {
        a1::Fork<animateTask>(a1::SetStaticSched(1,1,Sched::PartitionTask::SUBGRAPH))();
        a1::Fork<visuTask>()();

    }
};






#ifdef SOFA_SMP_WEIGHT
SMPBglSimulation::SMPBglSimulation():visualNode(NULL),
    parallelCompile( initData( &parallelCompile, false, "parallelCompile", "Compile task graph in parallel"))
#else
SMPBglSimulation::SMPBglSimulation():visualNode(NULL)
#endif
{
    changeListener=new common::ChangeListener();

    multiGraph= new Iterative::Multigraph<MainLoopTask>();

    //-------------------------------------------------------------------------------------------------------
    sofa::core::ObjectFactory::AddAlias("DefaultCollisionGroupManager",
            "BglCollisionGroupManager", true, 0);

    sofa::core::ObjectFactory::AddAlias("CollisionGroup",
            "BglCollisionGroupManager", true, 0);

#ifdef SOFA_SMP_WEIGHT
    multiGraph2= new Iterative::Multigraph<MainLoopTask>();
    multiGraph2->deployed=true;
    multiGraph->deployed=true;
    multiGraph2->compiled=true;
    multiGraph->compiled=true;
#endif
}

SMPBglSimulation::~SMPBglSimulation()
{
}

void SMPBglSimulation::init( Node* root )
{
    Simulation::init(root);
    BglGraphManager::getInstance()->update();
    changeListener->addChild ( NULL,  dynamic_cast< Node *>(root) );
}

/// Create a graph node and attach a new Node to it, then return the Node
Node* SMPBglSimulation::createNewGraph(const std::string& name)
{
    return new BglNode(name);
}

/**
Data: hgraph, rgraph
Result: hroots, interaction groups, all nodes initialized.
*/



/// Create a Node bgl structure using available file loaders, then convert it to a SMPBglSimulation
Node* SMPBglSimulation::load(const char* f)
{
    Node *root=Simulation::load(f);
    BglGraphManager::getInstance()->update();
    return root;

    //Temporary: we need to change that: We could change getRoots by a getRoot.
    //if several roots are found, we return a master node, above the roots of the simulation

    //         std::vector< Node* > roots;
    //         BglGraphManager::getInstance()->getRoots(roots);
    //         if (roots.empty()) return NULL;
    //         return roots.back();
}


Node *SMPBglSimulation::getVisualRoot()
{
    return BglGraphManager::getInstance()->getVisualRoot();
}

void SMPBglSimulation::reset(Node* root)
{
    sofa::simulation::Simulation::reset(root);
    BglGraphManager::getInstance()->reset();
}

void SMPBglSimulation::unload(Node* root)
{
    BglNode *n=dynamic_cast<BglNode*>(root);
    if (!n) return;
    helper::vector< Node* > parents;
    n->getParents(parents);
    if (parents.empty()) //Root
    {
        Simulation::unload(getVisualRoot());
    }
}

// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
void SMPBglSimulation::animate ( Node* root, double dt )
{
    if ( !root ) return;

    _root=root;
    _dt=dt;

    double nextTime = root->getTime() + root->getDt();

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        root->execute ( act );
    }

    BehaviorUpdatePositionVisitor beh(_root->getDt());
    _root->execute ( beh );

    if (changeListener->changed()||nbSteps.getValue()<2)
    {

        sofa::core::CallContext::ProcessorType
        etype=sofa::core::CallContext::getExecutionType();
        sofa::core::CallContext::setExecutionType(sofa::core::CallContext::GRAPH_KAAPI);

        multiGraph->compile();

        sofa::core::CallContext::setExecutionType(etype);

        multiGraph->deploy();

        changeListener->reset();
    }

    multiGraph->step();
    _root->execute<CollisionVisitor>();

    _root->setTime(nextTime);
    _root->execute<VisualUpdateVisitor>();
    _root->execute<UpdateSimulationContextVisitor>();
    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        root->execute ( act );
    }
    *(nbSteps.beginEdit()) = nbSteps.getValue() + 1;
    nbSteps.endEdit();

}
void SMPBglSimulation::generateTasks ( Node* root, double dt )
{
    if ( !root ) return;
    if ( root->getMultiThreadSimulation() )
        return;

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        root->execute ( act );
    }

    double startTime = root->getTime();
    double mechanicalDt = dt/numMechSteps.getValue();

    // CHANGE to support MasterSolvers : CollisionVisitor is now activated within AnimateVisitor
    //root->execute<CollisionVisitor>();

    AnimateVisitor act;
    act.setDt ( mechanicalDt );
    BehaviorUpdatePositionVisitor beh(root->getDt());
    for ( unsigned i=0; i<numMechSteps.getValue(); i++ )
    {
        root->execute ( act );
        root->execute ( beh );
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
    root->execute<ParallelVisualUpdateVisitor>();
    root->execute<ParallelCollisionVisitor>();
}

}
}
}


