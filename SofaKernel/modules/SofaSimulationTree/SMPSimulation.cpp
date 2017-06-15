/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaSimulationTree/SMPSimulation.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/PrintVisitor.h>
#include <SofaSimulationCommon/FindByTypeVisitor.h>
#include <sofa/simulation/ExportGnuplotVisitor.h>
#include <sofa/simulation/InitVisitor.h>
#include <sofa/simulation/AnimateVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/ResetVisitor.h>
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/ExportOBJVisitor.h>
#include <sofa/simulation/WriteStateVisitor.h>
#include <sofa/simulation/XMLPrintVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/core/ObjectFactory.h>
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

namespace tree
{

Node *_root=NULL;
double _dt;
struct doCollideTask
{
    void operator()(const sofa::core::ExecParams *params)
    {
        _root->execute<CollisionVisitor>(params);

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

    void operator()(const sofa::core::ExecParams *params)
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
        ctime_t t1 = CTime::getRefTime ();
        std::cerr << "Compiling Time: " << ((t1 - t0) / (CTime::getRefTicksPerSec () / 1000)) * 0.001<< std::endl;

    }
};
#endif

struct animateTask
{
    void operator()(const sofa::core::ExecParams* /* *params */)
    {
        sofa::simulation::tree::SMPSimulation* simulation=dynamic_cast<SMPSimulation *>(sofa::simulation::tree::getSimulation());
        simulation->generateTasks( _root);
#ifdef SOFA_SMP_WEIGHT
        a1::Fork<compileGraphTask>(a1::SetSite(2))();
#endif

    }
};

struct collideTask
{
    void operator()(const sofa::core::ExecParams *params)
    {

        a1::Fork<doCollideTask>()(params);

    }
};
struct visuTask
{
    void operator()(const sofa::core::ExecParams* /**params*/)
    {
        //ColisonBeginEvent!!
        //_root->execute<CollisionVisitor>();
    }
};
struct MainLoopTask
{

    void operator()()
    {
        sofa::core::ExecParams* params = new sofa::core::ExecParams();
        params->setExecMode(sofa::core::ExecParams::EXEC_KAAPI);
        a1::Fork<animateTask>(a1::SetStaticSched(1,1,Sched::PartitionTask::SUBGRAPH))(params);
        a1::Fork<visuTask>()(params);

    }
};


#ifdef SOFA_SMP_WEIGHT
SMPSimulation::SMPSimulation():visualNode(NULL),
    parallelCompile( initData( &parallelCompile, false, "parallelCompile", "Compile task graph in parallel"))
#else
SMPSimulation::SMPSimulation():visualNode(NULL)
#endif
{
    changeListener=new common::ChangeListener();

    multiGraph= new Iterative::Multigraph<MainLoopTask>();
#ifdef SOFA_SMP_WEIGHT
    multiGraph2= new Iterative::Multigraph<MainLoopTask>();
    multiGraph2->deployed=true;
    multiGraph->deployed=true;
    multiGraph2->compiled=true;
    multiGraph->compiled=true;
#endif
    sofa::core::ObjectFactory::AddAlias("CGLinearSolver","ParallelCGLinearSolver",true, 0);
    sofa::core::ObjectFactory::AddAlias("CGSolver","ParallelCGLinearSolver",true,0);
    sofa::core::ObjectFactory::AddAlias("ConjugateGradient","ParallelCGLinearSolver",true,0);
}

SMPSimulation::~SMPSimulation()
{
}
void SMPSimulation::init( Node* root )
{
    Simulation::init(root);
    changeListener->addChild ( NULL,  dynamic_cast< GNode *>(root) );
}

Node::SPtr SMPSimulation::createNewGraph(const std::string& name)
{
    sRoot = sofa::core::objectmodel::New<GNode>(name);
    return sRoot;
}

Node::SPtr SMPSimulation::createNewNode(const std::string& name)
{
    return sofa::core::objectmodel::New<GNode>(name);
}


// Execute one timestep. If dt is 0, the dt parameter in the graph will be used
void SMPSimulation::animate ( Node* root, double dt )
{
    if ( !root ) return;

    sofa::core::ExecParams* params = new sofa::core::ExecParams();
    params->setExecMode(sofa::core::ExecParams::EXEC_KAAPI);

    _root=root;
    _dt=dt;

    double nextTime = root->getTime() + root->getDt();

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params /* PARAMS FIRST */, &ev );
        root->execute ( act );
    }

    BehaviorUpdatePositionVisitor beh(params /* PARAMS FIRST */, _root->getDt());
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
    _root->execute<CollisionVisitor>(params);

    _root->setTime(nextTime);
    _root->execute<VisualUpdateVisitor>(params);
    _root->execute<UpdateSimulationContextVisitor>(params);
    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params /* PARAMS FIRST */, &ev );
        root->execute ( act );
    }
    *(nbSteps.beginEdit()) = nbSteps.getValue() + 1;
    nbSteps.endEdit();
}// SMPASimulation::animate

void SMPSimulation::generateTasks ( Node* root, double dt )
{
    if ( !root ) return;

    sofa::core::ExecParams* params = new sofa::core::ExecParams();
    params->setExecMode(sofa::core::ExecParams::EXEC_KAAPI);

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params /* PARAMS FIRST */, &ev );
        root->execute ( act );
    }

    //std::cout << "animate\n";
    double startTime = root->getTime();
    double mechanicalDt = dt/numMechSteps.getValue();
    //double nextTime = root->getTime() + root->getDt();

    // CHANGE to support AnimationLoop : CollisionVisitor is now activated within AnimateVisitor
    //root->execute<CollisionVisitor>();

    AnimateVisitor act( params );
    act.setDt ( mechanicalDt );
    BehaviorUpdatePositionVisitor beh(params /* PARAMS FIRST */, root->getDt());
    for ( unsigned i=0; i<numMechSteps.getValue(); i++ )
    {
        root->execute ( act );
        root->execute ( beh );
        root->setTime ( startTime + (i+1)* act.getDt() );
        root->execute<UpdateSimulationContextVisitor>(params);
    }

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params /* PARAMS FIRST */, &ev );
        root->execute ( act );
    }

    root->execute<UpdateMappingVisitor>(params);
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params /* PARAMS FIRST */, &ev );
        root->execute ( act );
    }
    root->execute<ParallelVisualUpdateVisitor>(params);
    root->execute<ParallelCollisionVisitor>(params);
}// SMPSimulation::generateTasks

Node *SMPSimulation::getVisualRoot()
{
    if (visualNode) return visualNode;
    else
    {
        visualNode= new GNode("VisualNode");
        visualNode->addTag(core::objectmodel::Tag("Visual"));
        return visualNode;
    }
}// SMPSimulation::getVisualRoot



SOFA_DECL_CLASS ( SMPSimulation );
// Register in the Factory
int SMPSimulationClass = core::RegisterObject ( "Main simulation algorithm" ) .add< SMPSimulation >();



} // namespace tree

} // namespace simulation

} // namespace sofa

