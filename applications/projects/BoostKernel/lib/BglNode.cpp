//
// C++ Implementation: BglNode
//
// Description:
//
//
// Author: Francois Faure in The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "BglNode.h"
#include "BglScene.h"
#include "bfs_adapter.h"
#include "dfs_adapter.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/vector_property_map.hpp>
#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{
namespace simulation
{
namespace bgl
{

BglNode::BglNode(BglScene* g, BglScene::Hvertex n, const std::string& name)
    : sofa::simulation::Node(name), scene(g), vertexId(n)
{
}


BglNode::~BglNode()
{
}


void BglNode::doExecuteVisitor( Visitor* vis )
{
    //cerr<<"BglNode::doExecuteVisitor( simulation::tree::Visitor* action)"<<endl;

    boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(scene->hgraph) );
    boost::queue<BglScene::Hvertex> queue;

    /*    boost::breadth_first_search(
            scene->hgraph,
            boost::vertex(this->vertexId, scene->hgraph),
            queue,
            bfs_adapter(vis,scene->h_vertex_node_map),
            colors
        );*/
    dfs_adapter dfs(vis,scene->h_vertex_node_map);
    boost::depth_first_visit(
        scene->hgraph,
        boost::vertex(this->vertexId, scene->hgraph),
        dfs,
        colors,
        dfs
    );
}

void BglNode::clearInteractionForceFields()
{
    interactionForceField.clear();
}


void BglNode::printComponents()
{
    using namespace sofa::core::componentmodel::behavior;
    using core::BaseMapping;
    using core::componentmodel::topology::Topology;
    using core::componentmodel::topology::BaseTopology;
    using core::componentmodel::topology::BaseMeshTopology;
    using core::Shader;
    using core::BehaviorModel;
    using core::VisualModel;
    using core::CollisionModel;
    using core::objectmodel::ContextObject;
    using core::componentmodel::collision::Pipeline;

    cerr<<"MasterSolver: ";
    for ( Single<MasterSolver>::iterator i=masterSolver.begin(), iend=masterSolver.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"OdeSolver: ";
    for ( Sequence<OdeSolver>::iterator i=solver.begin(), iend=solver.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"InteractionForceField: ";
    for ( Sequence<InteractionForceField>::iterator i=interactionForceField.begin(), iend=interactionForceField.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl<<"ForceField: ";
    for ( Sequence<BaseForceField>::iterator i=forceField.begin(), iend=forceField.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"State: ";
    for ( Single<BaseMechanicalState>::iterator i=mechanicalState.begin(), iend=mechanicalState.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"Mechanical Mapping: ";
    for ( Single<BaseMechanicalMapping>::iterator i=mechanicalMapping.begin(), iend=mechanicalMapping.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"Mapping: ";
    for ( Sequence<BaseMapping>::iterator i=mapping.begin(), iend=mapping.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"Topology: ";
    for ( Single<Topology>::iterator i=topology.begin(), iend=topology.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"BaseTopology: ";
    for ( Sequence<BaseTopology>::iterator i=basicTopology.begin(), iend=basicTopology.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"MeshTopology: ";
    for ( Single<BaseMeshTopology>::iterator i=meshTopology.begin(), iend=meshTopology.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"Shader: ";
    for ( Single<Shader>::iterator i=shader.begin(), iend=shader.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"Constraint: ";
    for ( Sequence<BaseConstraint>::iterator i=constraint.begin(), iend=constraint.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"BehaviorModel: ";
    for ( Sequence<BehaviorModel>::iterator i=behaviorModel.begin(), iend=behaviorModel.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"VisualModel: ";
    for ( Sequence<VisualModel>::iterator i=visualModel.begin(), iend=visualModel.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"CollisionModel: ";
    for ( Sequence<CollisionModel>::iterator i=collisionModel.begin(), iend=collisionModel.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"ContextObject: ";
    for ( Sequence<ContextObject>::iterator i=contextObject.begin(), iend=contextObject.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"Pipeline: ";
    for ( Single<Pipeline>::iterator i=collisionPipeline.begin(), iend=collisionPipeline.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
    cerr<<endl<<"VisitorScheduler: ";
    for ( Single<VisitorScheduler>::iterator i=actionScheduler.begin(), iend=actionScheduler.end(); i!=iend; i++ )
        cerr<<(*i)->getName()<<" ";
    cerr<<endl;
}

}
}
}


