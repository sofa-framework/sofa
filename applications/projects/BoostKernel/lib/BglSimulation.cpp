/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
 *                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
 *                                                                             *
 * This program is free software; you can redistribute it and/or modify it     *
 * under the terms of the GNU General Public License as published by the Free  *
 * Software Foundation; either version 2 of the License, or (at your option)   *
 * any later version.                                                          *
 *                                                                             *
 * This program is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
 * more details.                                                               *
 *                                                                             *
 * You should have received a copy of the GNU General Public License along     *
 * with this program; if not, write to the Free Software Foundation, Inc., 51  *
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
 *******************************************************************************
 *                            SOFA :: Applications                             *
 *                                                                             *
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
//
// C++ Implementation: BglSimulation
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
#include "BglSimulation.h"
#include "dfs_adapter.h"
#include "dfv_adapter.h"
#include "BuildNodesFromGNodeVisitor.h"
#include "BuildRestFromGNodeVisitor.h"

#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/vector_property_map.hpp>

#include <sofa/simulation/tree/TreeSimulation.h>

#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/PrintVisitor.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/VisualVisitor.h>

#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>

#include <sofa/helper/system/FileRepository.h>

#include <iostream>
using std::cerr;
using std::endl;

namespace sofa
{
namespace simulation
{
namespace bgl
{


Simulation* getSimulation()
{
    if ( Simulation::Simulation::theSimulation==NULL )
        setSimulation(new BglSimulation);
    return simulation::getSimulation();
}

BglSimulation::BglSimulation():collisionPipeline(NULL)
{
    h_vertex_node_map = get( bglnode_t(), hgraph);
    r_vertex_node_map = get( bglnode_t(), rgraph);
    //c_vertex_node_map = get( bglnode_t(), cgraph);

    // The animation control overloads the solvers of the scene
    masterNode= static_cast<BglNode*>(newNode("masterNode"));
    masterVertex = h_node_vertex_map[masterNode];
}

// BglSimulation::Hvertex BglSimulation::addHvertex( sofa::simulation::Node* node )
// {
//     Hvertex hvertex =  add_vertex( hgraph);
//     h_vertex_node_map[hvertex] = node;
//      h_node_vertex_map[node] = hvertex;
//
//      Rvertex rvertex =  add_vertex( rgraph);
//     r_vertex_node_map[rvertex] = node;
//      r_node_vertex_map[node] = rvertex;
//
//      return node;
// }

BglSimulation::Hedge BglSimulation::addHedge( Hvertex p, Hvertex c )
{
    std::pair<Hedge, bool> e =  add_edge(  p,c,hgraph);
    assert(e.second);
    return e.first;
}

BglSimulation::Hedge BglSimulation::addRedge( Hvertex p, Hvertex c )
{
    std::pair<Hedge, bool> e =  add_edge(  p,c,rgraph);
    assert(e.second);
    return e.first;
}

void BglSimulation::removeHedge( Hvertex p, Hvertex c )
{
    remove_edge(  p,c,hgraph);
}

void BglSimulation::removeRedge( Hvertex p, Hvertex c )
{
    remove_edge(  p,c,rgraph);
}


/// Create a graph node and attach a new Node to it, then return the Node
Node* BglSimulation::newNode(const std::string& name)
{
    // Each BglNode needs a vertex in hgraph
    Hvertex hnode =  add_vertex( hgraph);
    BglNode* s  = new BglNode(this,&hgraph,hnode,name);
    h_vertex_node_map[hnode] = s;
    h_node_vertex_map[s] = hnode;
    // add it to rgraph
    Rvertex rnode = add_vertex( rgraph );
    r_vertex_node_map[rnode] = s;
    r_node_vertex_map[s] = rnode;
    return s;
}

///
void BglSimulation::setMechanicalMapping(BglNode* p, BglNode* c, MechanicalMapping* m )
{
    addHedge( h_node_vertex_map[p], h_node_vertex_map[c] );
    addRedge( h_node_vertex_map[c], h_node_vertex_map[p] );
    c->addObject(m);
}

void BglSimulation::addInteraction( Node* n1, Node* n2, InteractionForceField* iff )
{
    interactions.push_back( Interaction(h_node_vertex_map[n1], h_node_vertex_map[n2], iff) );
}

namespace
{

class find_leaves: public ::boost::bfs_visitor<>
{
public:
    typedef vector<BglSimulation::Rvertex> Rleaves;
    Rleaves& leaves; // use external data, since internal data seems corrupted after the visit (???)
    find_leaves( Rleaves& l ):leaves(l) {}
    void discover_vertex( BglSimulation::Rvertex v, const BglSimulation::Rgraph& g )
    {
        if ( out_degree (v,g)==0 )  // leaf vertex
        {
            leaves.push_back(v);
        }
    }

};
}

/**
Data: hroots, interactions
 Result: interactionGroups
    */
void BglSimulation::computeInteractionGraphAndConnectedComponents()
{
    i_vertex_node_map = get( bglnode_t(), igraph);
    i_edge_interaction_map = get( interaction_t(), igraph );

    // create the interaction graph vertices (root nodes only)
    igraph.clear();
    for ( HvertexVector::iterator i=hroots.begin(), iend=hroots.end(); i!=iend; i++ )
    {
        Ivertex iv = add_vertex( igraph );
        Node* n = h_vertex_node_map[*i];
        i_vertex_node_map[iv] = n;
        i_node_vertex_map[n] = iv;
    }

    // 	  cerr<<"interaction nodes: "<<endl;
    // 	    for( Ivpair i = vertices(igraph); i.first!=i.second; i.first++ )
    // 	    cerr<<i_vertex_node_map[*i.first]->getName()<<", ";
    // 	    cerr<<endl;
    // 	    cerr<<"begin create interaction edges"<<endl;

    // create the edges between the root nodes and associate the interactions with the root nodes
    // rgraph is used to find the roots corresponding to the nodes.
    typedef std::map<Rvertex,Interactions > R_vertex_interactions_map;
    R_vertex_interactions_map rootInteractions;
    for ( Interactions::iterator i=interactions.begin(), iend=interactions.end(); i!=iend; i++ )
    {
        // 	    cerr<<"find all the roots associated with the interaction from "<<h_vertex_node_map[(*i).v1]->getName()<<" to "<<h_vertex_node_map[(*i).v2]->getName()<<endl;

        // find all the roots associated with the given interaction
        vector<BglSimulation::Rvertex> leaves;
        find_leaves visit(leaves);
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(rgraph) );
        boost::queue<Rvertex> queue;
        Rvertex v1 = r_node_vertex_map[ h_vertex_node_map[ (*i).v1 ]];
        Rvertex v2 = r_node_vertex_map[ h_vertex_node_map[ (*i).v2 ]];
        boost::breadth_first_search( rgraph, boost::vertex(v1, rgraph), queue, visit, colors );
        boost::breadth_first_search( rgraph, boost::vertex(v2, rgraph), queue, visit, colors );

        //             cerr<<"the roots are: "<<endl;
        //             for( unsigned j=0; j<leaves.size(); j++ ){
        //               cerr<<r_vertex_node_map[visit.leaves[j]]->getName()<<", ";
        //             }
        //             cerr<<endl;

        // associate the interaction with one of its roots, no matter which one because it will then be associated to the whole interaction group.
        assert( visit.leaves.size()>0 );
        rootInteractions[*visit.leaves.begin()].push_back( *i );

        // add edges between all the pairs of roots
        for ( find_leaves::Rleaves::iterator l=visit.leaves.begin(), lend=visit.leaves.end(); l!=lend; l++ )
        {
            for ( find_leaves::Rleaves::iterator m=l++; m!=lend; m++ )
            {
                std::pair<Iedge,bool> e = add_edge( i_node_vertex_map[r_vertex_node_map[*l]],i_node_vertex_map[r_vertex_node_map[*m]], igraph );
                assert( e.second );
            }
        }
    }

    // compute the connected components of the interaction graph, represented by integers associated with vertices
    vector<int> component(num_vertices(igraph));
    int num = boost::connected_components(igraph, &component[0]);

    // build the interactionGroups
    interactionGroups.clear();
    interactionGroups.resize(num);
    int index = 0;
    for ( Ivpair i=vertices(igraph); i.first!=i.second; i.first++,index++ )
    {
        Ivertex iv = *i.first;
        Hvertex hv = h_node_vertex_map[ i_vertex_node_map[iv]]; // hv(iv)
        Rvertex rv = r_node_vertex_map[ i_vertex_node_map[iv]]; // rv(iv)
        InteractionGroup& group = interactionGroups[component[index]]; // the group the node belongs to

        group.first.push_back( hv );    // add the node to its interaction group
        // add its associated interaction to the interaction group
        for ( Interactions::iterator j=rootInteractions[rv].begin(), jend=rootInteractions[rv].end(); j!=jend; j++ )
        {
            group.second.push_back( *j );
        }
    }
    // 	debug
    // 	     cerr<<"end connected components"<<endl;
    for( unsigned i=0; i<interactionGroups.size(); i++ )
    {
        cerr<<"interaction group (roots only): "<<endl;
        cerr<<"- nodes = ";
        for( unsigned j=0; j<interactionGroups[i].first.size(); j++ )
        {
            Node* root = h_vertex_node_map[ interactionGroups[i].first[j] ];
            cerr<< root->getName() <<", ";
        }
        cerr<<endl<<"- interactions = ";
        for( unsigned j=0; j<interactionGroups[i].second.size(); j++ )
        {
            Node* n1 = r_vertex_node_map[ interactionGroups[i].second[j].v1 ];
            Node* n2 = r_vertex_node_map[ interactionGroups[i].second[j].v2 ];
            InteractionForceField* iff = interactionGroups[i].second[j].iff;
            cerr<<iff->getName()<<" between "<<n1->getName()<<" and "<<n2->getName()<< ", ";
        }
        cerr<<endl;
    }

}

/**
Data: hgraph, rgraph
 Result: hroots, interaction groups, all nodes initialized.
    */
void BglSimulation::init()
{
    cerr<<"begin BglSimulation::init()"<<endl;

    /// find the roots in hgraph
    hroots.clear();
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
    {
        if ( *iter.first != masterVertex && in_degree (*iter.first,hgraph)==0 )
        {
            hroots.push_back(*iter.first);
            //cerr<<"node "<<h_vertex_node_map[*iter.first]->getName()<<" is a root"<<endl;
        }
    }

    /// Initialize all the nodes using a depth-first visit
    for (HvertexVector::iterator i=hroots.begin(), iend=hroots.end(); i!=iend; i++ )
    {
        h_vertex_node_map[*i]->init(); // init the root and its subgraph
        //Link all the nodes to the MasterNode, in ordre to propagate the Context
        addHedge( masterVertex, *i); // add nodes
    }
    masterNode->init(); // not in hroots
    /// compute the interaction groups
    computeInteractionGraphAndConnectedComponents();

    /// initialize the visual models and their mappings
    for ( VisualVector::iterator i=visualModels.begin(), iend=visualModels.end(); i!=iend; i++ )
    {
        //cerr<<"init visual model, mapped from "<<(*i).second->getFrom()->getName()<<" to "<<(*i).second->getTo()->getName()<<endl;
        (*i).first->init();
        (*i).second->init();
    }




    /// Put the collision models in the collision graph
    /*      for( Hvpair v = vertices(hgraph); v.first!=v.second; v.first++ )
    	{
                BglNode* node = h_vertex_node_map[*v.first];
                if( !node->collisionModel.empty() )
                {
    	// insert the node in the collision graph
    	Hvertex cvertex = add_vertex( cgraph );
    	c_vertex_node_map[cvertex] = node;
    	cerr<<"add "<<node->getName()<<" to the collision graph"<<endl;
                }
    	}*/



    //cerr<<"end BglSimulation::init()"<<endl;

    // debug
    //      cerr<<"there are "<<num_vertices(hgraph)<<" hvertices:"<<endl;
    //      for( Hvpair i=vertices(hgraph); i.first!=i.second; i.first++ )
    //      {
    //              cerr<<h_vertex_node_map[*i.first]->getName()<<", ";
    //      }
    //      cerr<<endl;
    //      cerr<<"there are "<<num_vertices(rgraph)<<" rvertices:"<<endl;
    //      for( Rvpair i=vertices(rgraph); i.first!=i.second; i.first++ )
    //      {
    //              cerr<<r_vertex_node_map[*i.first]->getName()<<", ";
    //      }
    //      cerr<<endl;

}

/** Data: interaction groups
Result: nodes updated
*/

void BglSimulation::mechanicalStep(Node* root, double dt)
{
    /** Put everyone in the graph to apply collision detection */
    //         cerr<<"BglSimulation::animate, start collision detection"<<endl;
    if (collisionPipeline)
    {
        masterNode->addObject( collisionPipeline );

        for (HvertexVector::iterator i=hroots.begin(), iend=hroots.end(); i!=iend; i++ )
        {
            add_edge( masterVertex, *i, hgraph );
            //                 cerr<<"BglSimulation::animate, adding node "<<h_vertex_node_map[*i]->getName()<<endl;
        }


        CollisionVisitor act;
        masterNode->doExecuteVisitor(&act);

        // collision detection was performed.
        // todo: update interaction groups
        // currently, we study the problem of a moving object against a fixed one, so interaction groups are not yet necessary

        // remove the pipeline, so that animation only integrates time
        masterNode->removeObject( collisionPipeline );
    }
    /** Update each interaction group independently.
        The master node is used to process all of them sequentially, but several copies of the master node would allow paralle processing.
    */
    //         std::cerr << interactionGroups.size() << " interactions \n";


    for ( unsigned i=0; i<interactionGroups.size(); i++ )
    {

        // remove previous children and interactions
        clear_vertex( masterVertex, hgraph );
        masterNode->clearInteractionForceFields();



        // add the vertices and the interactions
        InteractionGroup& group = interactionGroups[i];
        std::vector< BglSimulation::Hedge > listHedgeCreated;
        Node *solverNode=masterNode;
        for ( unsigned j=0; j<group.first.size(); j++ )
        {
            solverNode = node_solver_map[ h_vertex_node_map[ group.first[j] ] ];
            if (!solverNode)
            {
                addHedge( masterVertex, group.first[j]); // add nodes
                solverNode = masterNode;
            }
            else
            {
                BglSimulation::Hedge e=addHedge( h_node_vertex_map[solverNode], group.first[j]); // add nodes
                listHedgeCreated.push_back(e);
                addHedge( masterVertex, h_node_vertex_map[solverNode]); // add nodes
            }
        }

        for ( unsigned j=0; j<group.second.size(); j++ )
        {
            InteractionForceField* iff = group.second[j].iff;
            solverNode->addObject( iff ); // add interactions
        }

#ifdef DUMP_VISITOR_INFO
        simulation::Visitor::printComment(std::string("Animate ") + h_vertex_node_map[ group.first[0] ]->getName() );
#endif
        // animate this interaction group
        masterNode->animate(dt);

        // remove the interaction forcefields
        for ( unsigned j=0; j<group.second.size(); j++ )
        {
            InteractionForceField* iff = group.second[j].iff;
            solverNode->removeObject( iff ); // add interactions
        }
        // remove the edges added to the graph
        for (unsigned int i=0; i<listHedgeCreated.size(); ++i)
            remove_edge( listHedgeCreated[i], hgraph);

    }
}

/// TODO: adapt the AnimateVisitor to BGL
void BglSimulation::animate(Node* root, double dt)
{
    dt = root->getContext()->getDt();

#ifdef DUMP_VISITOR_INFO
    simulation::Visitor::printComment(std::string("Begin Step"));
#endif
    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        masterNode->doExecuteVisitor ( &act );
    }


    double startTime = root->getTime();
    double mechanicalDt = dt/numMechSteps.getValue();

    BehaviorUpdatePositionVisitor beh(dt);

    for( unsigned step=0; step<numMechSteps.getValue(); step++ )
    {
        masterNode->doExecuteVisitor ( &beh );
        mechanicalStep(root,dt);
        masterNode->setTime ( startTime + (step+1)* mechanicalDt );
    }

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( &ev );
        masterNode->doExecuteVisitor( &act );
    }
#ifdef DUMP_VISITOR_INFO
    simulation::Visitor::printComment(std::string("End Step"));
#endif


}

void BglSimulation::computeBBox(Node* root, SReal* minBBox, SReal* maxBBox)
{
    /// TODO: use root only
    /// TODO: add mechanical object
    for ( VisualVector::iterator i=visualModels.begin(), iend=visualModels.end(); i!=iend; i++ )
    {
        (*i).first->addBBox(minBBox,maxBBox);
    }

}

/// Toggle the showBehaviorModel flag of all systems
void BglSimulation::setShowBehaviorModels( bool b )
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowBehaviorModels(b);
}
/// Toggle the showVisualModel flag of all systems
void BglSimulation::setShowVisualModels( bool b )
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowVisualModels(b);
}
/// Toggle the showNormals flag of all systems
void BglSimulation::setShowNormals( bool b )
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowNormals(b);
}

/// Display flags: Collision Models
void BglSimulation::setShowCollisionModels(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowCollisionModels(val);
}

/// Display flags: Bounding Collision Models
void BglSimulation::setShowBoundingCollisionModels(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowBoundingCollisionModels(val);
}

/// Display flags: Mappings
void BglSimulation::setShowMappings(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowMappings(val);
}

/// Display flags: Mechanical Mappings
void BglSimulation::setShowMechanicalMappings(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowMechanicalMappings(val);
}

/// Display flags: ForceFields
void BglSimulation::setShowForceFields(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowForceFields(val);
}

/// Display flags: InteractionForceFields
void BglSimulation::setShowInteractionForceFields(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowInteractionForceFields(val);
}

/// Display flags: WireFrame
void BglSimulation::setShowWireFrame(bool val)
{
    for ( Hvpair iter=boost::vertices(hgraph); iter.first!=iter.second; iter.first++)
        h_vertex_node_map[*iter.first]->setShowWireFrame(val);
}

void BglSimulation::setVisualModel( VisualModel* v, Mapping* m)
{
    visualModels.push_back( std::make_pair(v,m) );
}

void BglSimulation::draw(Node* , helper::gl::VisualParameters*)
{
    // 	cerr<<"begin BglSimulation::glDraw()"<<endl;
    for (HvertexVector::iterator i=hroots.begin(), iend=hroots.end(); i!=iend; i++ )
        h_vertex_node_map[*i]->glDraw();

    for ( VisualVector::iterator i=visualModels.begin(), iend=visualModels.end(); i!=iend; i++ )
    {
        //cerr<<"draw visual model, mapped from "<<(*i).second->getFrom()->getName()<<" to "<<(*i).second->getTo()->getName()<<endl;
        (*i).second->updateMapping();
        (*i).first->drawVisual();
    }
    // 	cerr<<"end BglSimulation::glDraw()"<<endl;
}


/// Create a GNode tree structure using available file loaders, then convert it to a BglSimulation
Node* BglSimulation::load(const char* f)
{
    std::string fileName(f);
    /*  if (fileName.empty())*/
    {
        //         fileName = "liver.scn";
        sofa::helper::system::DataRepository.findFile(fileName);
    }

    std::cerr << "Loading " << f << "\n";

    sofa::simulation::tree::GNode* groot = 0;

    sofa::simulation::tree::TreeSimulation treeSimu;
    std::string in_filename(fileName);
    if (in_filename.rfind(".simu") == std::string::npos)
        groot = dynamic_cast< sofa::simulation::tree::GNode* >(treeSimu.load(fileName.c_str()));

    std::cerr << "File Loaded\n";
    if ( !groot )
    {
        cerr<<"BglSimulation::load file "<<fileName<<" failed"<<endl;
        exit(1);
    }
    //else cerr<<"BglSimulation::loaded file "<<fileName<<endl;

    //      cerr<<"GNode loaded: "<<endl;
    //      groot->execute<PrintVisitor>();
    //      cerr<<"==========================="<<endl;
    std::map<simulation::Node*,BglNode*> gnode_bnode_map;
    BuildNodesFromGNodeVisitor b1(this);
    groot->execute(b1);
    gnode_bnode_map = b1.getGNodeBNodeMap();
    BuildRestFromGNodeVisitor b2(this);
    b2.setGNodeBNodeMap(gnode_bnode_map);
    groot->execute(b2);

    init();
    return masterNode;

    /*    cerr<<"loaded graph has "<<num_vertices(hgraph)<<" vertices and "<<num_edges(hgraph)<<" edges:"<<endl;
          PrintVisitor printvisitor;
          dfs( printvisitor );*/
}
/// Add an OdeSolver working inside a given Node
void BglSimulation::addSolver(BaseObject* s,Node* n)
{
    if (!n) masterNode->moveObject(s);
    else
    {
        Node *solverNode = node_solver_map[n];
        if (!solverNode) //Node not created yet
        {
            std::ostringstream s;
            s << "SolverNode" << node_solver_map.size();
            solverNode = newNode(s.str());
            node_solver_map[n]=solverNode;
            addHedge( masterVertex, h_node_vertex_map[solverNode]);
        }
        std::cerr << "Adding Solver : " << s->getName() << " To " << solverNode->getName() << "\n";
        solverNode->moveObject(s);
    }
}



void BglSimulation::unload(Node* root)
{
    std::cerr << "UNLOAD TO IMPLEMENT FOR BGL: " << root << "\n";
}

/// depth search in the whole scene
void BglSimulation::dfs( Visitor& visit )
{
    dfs_adapter vis(&visit,h_vertex_node_map);
    boost::depth_first_search( hgraph, boost::visitor(vis) );
}
/// depth search starting from the given vertex, and prunable
void BglSimulation::dfv( Hvertex v, Visitor& vis )
{
    boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(hgraph) );
    dfv_adapter dfsv(&vis,h_vertex_node_map );
    boost::depth_first_visit(
        hgraph,
        boost::vertex(v, hgraph),
        dfsv,
        colors,
        dfsv
    );
}



}
}
}


