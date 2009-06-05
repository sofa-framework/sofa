/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include "GetObjectsVisitor.h"


#include <sofa/core/objectmodel/BaseContext.h>
//Components of the core to detect during the addition of objects in a node
#include <sofa/core/componentmodel/behavior/BaseMechanicalMapping.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/componentmodel/behavior/InteractionForceField.h>
#include <sofa/core/componentmodel/behavior/InteractionConstraint.h>
#include <sofa/core/componentmodel/behavior/MasterSolver.h>
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>

#include <sofa/simulation/common/AnimateVisitor.h>


//#include "bfs_adapter.h"
#include "dfv_adapter.h"
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/topological_sort.hpp>
//#include <boost/property_map.hpp>
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

BglNode::BglNode(BglGraphManager* s,const std::string& name)
    : sofa::simulation::Node(name), graphManager(s), graph(NULL)
{

}

BglNode::BglNode(BglGraphManager* s, BglGraphManager::Hgraph *g,  BglGraphManager::Hvertex n, const std::string& name)
    : sofa::simulation::Node(name), graphManager(s), graph(g), vertexId(n)
{

}

BglNode::~BglNode()
{
    //std::cerr << "Destruction of Node this : " << this->getName() << " ; " << graphManager->h_node_vertex_map[this] << " DELETED\n";
    graphManager->deleteNode(this);
}


/// Do one step forward in time
void BglNode::animate( double dt )
{
    simulation::AnimateVisitor vis(dt);
    //cerr<<"Node::animate, start execute"<<endl;
    doExecuteVisitor(&vis);
    //cerr<<"Node::animate, end execute"<<endl;
}

bool BglNode::addObject(BaseObject* obj)
{
    if (sofa::core::componentmodel::behavior::BaseMechanicalMapping* mm = dynamic_cast<sofa::core::componentmodel::behavior::BaseMechanicalMapping*>(obj))
    {
        if (mm->getMechFrom() == NULL) {std::cerr << "ERROR in addObject BglNode: RayPick Issue!!\n"; return false; }
        Node *from=(Node*)mm->getMechFrom()->getContext();
        Node *to=(Node*)mm->getMechTo()->getContext();
        graphManager->addEdge(from, to);
    }
    else if (sofa::core::componentmodel::behavior::InteractionForceField* iff = dynamic_cast<sofa::core::componentmodel::behavior::InteractionForceField*>(obj))
    {
        graphManager->addInteraction( (Node*)iff->getMechModel1()->getContext(),
                (Node*)iff->getMechModel2()->getContext(),
                iff);
    }
    else if (sofa::core::componentmodel::behavior::InteractionConstraint* ic = dynamic_cast<sofa::core::componentmodel::behavior::InteractionConstraint*>(obj))
    {
        graphManager->addInteraction( (Node*)ic->getMechModel1()->getContext(),
                (Node*)ic->getMechModel2()->getContext(),
                ic);
    }
    else if (sofa::core::componentmodel::behavior::MasterSolver* ms = dynamic_cast<sofa::core::componentmodel::behavior::MasterSolver*>(obj))
    {
        graphManager->addSolver(ms,this);
    }
    else if (sofa::core::componentmodel::behavior::OdeSolver* odes = dynamic_cast<sofa::core::componentmodel::behavior::OdeSolver*>(obj))
    {
        graphManager->addSolver(odes,this);
    }
    return Node::addObject(obj);
}

bool BglNode::removeObject(core::objectmodel::BaseObject* obj)
{
    if (sofa::core::componentmodel::behavior::BaseMechanicalMapping* mm = dynamic_cast<sofa::core::componentmodel::behavior::BaseMechanicalMapping*>(obj))
    {
        Node *from=(Node*)mm->getMechFrom()->getContext();
        Node *to=(Node*)mm->getMechTo()->getContext();

        graphManager->removeEdge(from, to);
        Node::removeObject(obj);
        return true;
    }
    else if (sofa::core::componentmodel::behavior::InteractionForceField* iff = dynamic_cast<sofa::core::componentmodel::behavior::InteractionForceField*>(obj))
    {
        graphManager->removeInteraction(iff);
    }
    else if (sofa::core::componentmodel::behavior::InteractionConstraint* ic = dynamic_cast<sofa::core::componentmodel::behavior::InteractionConstraint*>(obj))
    {
        graphManager->removeInteraction(ic);
    }
    return Node::removeObject(obj);
}

void BglNode::addChild(core::objectmodel::BaseNode* c)
{
    BglNode *childNode = static_cast< BglNode *>(c);
    //std::cerr << "addChild : of "<< this->getName() << "@" << this << " and " <<  c->getName() << "@" << c << "\n";

    notifyAddChild(childNode);
    child.add(childNode);
    childNode->parents.add(this);
    graphManager->addNode(this,childNode);
}

void BglNode::removeChild(core::objectmodel::BaseNode* c)
{
    BglNode *childNode = static_cast< BglNode *>(c);
    //std::cerr << "deleteChild : of "<< this->getName() << "@" << this << " and " <<  c->getName() << "@" << c << "\n";

    notifyRemoveChild(childNode);
    child.remove(childNode);
    childNode->parents.remove(this);
    graphManager->deleteNode(childNode);
}

void BglNode::moveChild(core::objectmodel::BaseNode* c)
{
    BglNode* childNode=dynamic_cast<BglNode*>(c);
    if (!childNode) return;

    if (childNode->parents.empty())
    {
        addChild(childNode);
    }
    else
    {
        for (ParentIterator it = parents.begin(); it != parents.end(); it++)
        {
            BglNode *prev = *it;
            notifyMoveChild(childNode,prev);
            prev->removeChild(childNode);
        }
        addChild(childNode);
    }
}

void BglNode::detachFromGraph()
{
    const helper::vector< BglNode* > &parents = getParents();
    for (unsigned int i=0; i<parents.size(); ++i) parents[i]->removeChild(this);
}


/// Find all the Nodes pointing
helper::vector< BglNode* > BglNode::getParents() const
{
    helper::vector< BglNode* > p;
    if (!graph) return p;
    BglGraphManager::Hgraph::in_edge_iterator in_i, in_end;
    //Find all in-edges from the graph
    for (tie(in_i, in_end) = boost::in_edges(vertexId, *graph); in_i != in_end; ++in_i)
    {
        BglGraphManager::Hedge e=*in_i;
        BglGraphManager::Hvertex src=source(e, *graph);
        p.push_back(static_cast<BglNode*>(graphManager->getNodeFromHvertex(src)));
    }
    return p;
}


std::string BglNode::getPathName() const
{

    std::string str;
    const helper::vector< BglNode* > &parents=getParents();
    if (!parents.empty()) str = parents[0]->getPathName();
    str += '/';
    str += getName();
    return str;

}

/// Get children nodes
sofa::helper::vector< core::objectmodel::BaseNode* >  BglNode::getChildren()
{
    helper::vector< core::objectmodel::BaseNode* > p;
    if (!graph) return p;
    BglGraphManager::Hgraph::out_edge_iterator out_i, out_end;
    //Find all in-edges from the graph
    for (tie(out_i, out_end) = boost::out_edges(vertexId, *graph); out_i != out_end; ++out_i)
    {
        BglGraphManager::Hedge e=*out_i;
        BglGraphManager::Hvertex src=target(e, *graph);
        Node *n=graphManager->getNodeFromHvertex(src);
        if (n) p.push_back(n);
    }
    return p;

}

/// Get a list of child node
const sofa::helper::vector< core::objectmodel::BaseNode* >  BglNode::getChildren() const
{
    helper::vector< core::objectmodel::BaseNode* > p;
    if (!graph) return p;
    BglGraphManager::Hgraph::out_edge_iterator out_i, out_end;
    //Find all in-edges from the graph
    for (tie(out_i, out_end) = boost::out_edges(vertexId, *graph); out_i != out_end; ++out_i)
    {
        BglGraphManager::Hedge e=*out_i;
        BglGraphManager::Hvertex src=target(e, *graph);
        Node *n=graphManager->getNodeFromHvertex(src);
        if (n) p.push_back(n);
    }
    return p;

}




/// Topology
core::componentmodel::topology::Topology* BglNode::getTopology() const
{
    // return this->topology;
    if (this->topology)
        return this->topology;
    else
        return get<core::componentmodel::topology::Topology>();
}

/// Mesh Topology (unified interface for both static and dynamic topologies)
core::componentmodel::topology::BaseMeshTopology* BglNode::getMeshTopology() const
{
    if (this->meshTopology)
        return this->meshTopology;
    else
        return get<core::componentmodel::topology::BaseMeshTopology>();
}

/// Shader
core::objectmodel::BaseObject* BglNode::getShader() const
{
    if (shader)
        return shader;
    else
        return get<core::Shader>();
}

/// Mechanical Degrees-of-Freedom
core::objectmodel::BaseObject* BglNode::getMechanicalState() const
{
    // return this->mechanicalModel;
    if (this->mechanicalState)
        return this->mechanicalState;
    else
        return get<core::componentmodel::behavior::BaseMechanicalState>();
}



void BglNode::doExecuteVisitor( Visitor* vis )
{
    //cerr<<"BglNode::doExecuteVisitor( simulation::tree::Visitor* action)"<<endl;
    if (!graph) return;
    boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(*graph) );
    //boost::queue<BglGraphManager::Hvertex> queue;

    /*    boost::breadth_first_search(
          graph,
          boost::vertex(this->vertexId, *graph),
          queue,
          bfs_adapter(vis,graphManager->h_vertex_node_map),
          colors
          );*/

    dfv_adapter dfv(vis,graphManager, graphManager->h_vertex_node_map);
    boost::depth_first_visit(
        *graph,
        boost::vertex(this->vertexId, *graph),
        dfv,
        colors,
        dfv
    );
}



/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BglNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
{
    if (graphManager->isNodeCreated(this)) return NULL;//std::cerr << "ERROR !!!!!\n";
    GetObjectVisitor getobj(class_info);
    getobj.setTags(tags);
    if ( dir == SearchDown )
    {
//              std::cerr << "Search Down ";
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(graphManager->hgraph) );
        dfv_adapter dfv( &getobj,  graphManager, graphManager->h_vertex_node_map );
        boost::depth_first_visit(
            graphManager->hgraph,
            boost::vertex(this->vertexId, graphManager->hgraph),
            dfv,
            colors,
            dfv
        );
    }
    else if (dir== SearchUp )
    {
//              std::cerr << "Search Up ";
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(graphManager->rgraph) );
        dfv_adapter dfv( &getobj, graphManager, graphManager->r_vertex_node_map );
        BglGraphManager::Rvertex thisvertex = graphManager->convertHvertex2Rvertex(this->vertexId);
        boost::depth_first_visit(
            graphManager->rgraph,
            boost::vertex(thisvertex, graphManager->rgraph),
            dfv,
            colors,
            dfv
        );
    }
    else if (dir== SearchRoot )
    {
//              std::cerr << "Search Root ";
        graphManager->dfv( graphManager->getMasterVertex(), getobj );
    }
    else    // Local
    {
//              std::cerr << "Search Local ";
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            void* result = class_info.dynamicCast(*it);
            if (result != NULL && (tags.empty() || (*it)->getTags().includes(tags)))
            {
//                     std::cerr << "Single Search : " << sofa::helper::gettypename((class_info)) << " result : " << result << "\n";
                return result;
            }
        }
    }
//          std::cerr << "Single Search : " << sofa::helper::gettypename(class_info) << " result : " << getobj.getObject() << "\n";
    return getobj.getObject();
}


/// Generic object access, given a path from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BglNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const
{
    std::cerr << "Single Search with path NOT IMPLEMENTED for " << sofa::helper::gettypename(class_info) << "\n";
    return NULL;
}



/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void BglNode::getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
{
//         std::cerr << "Search for " << sofa::helper::gettypename(class_info) << "\n";
    GetObjectsVisitor getobjs(class_info, container);
    getobjs.setTags(tags);
    if ( dir == SearchDown )
    {
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(graphManager->hgraph) );
        dfv_adapter dfv( &getobjs, graphManager, graphManager->h_vertex_node_map );
        boost::depth_first_visit(
            graphManager->hgraph,
            boost::vertex(this->vertexId, graphManager->hgraph),
            dfv,
            colors,
            dfv
        );
    }
    else if (dir== SearchUp )
    {
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(graphManager->rgraph) );
        dfv_adapter dfv( &getobjs, graphManager, graphManager->r_vertex_node_map );
        BglGraphManager::Rvertex thisvertex = graphManager->convertHvertex2Rvertex(this->vertexId);
        boost::depth_first_visit(
            graphManager->rgraph,
            boost::vertex(thisvertex, graphManager->rgraph),
            dfv,
            colors,
            dfv
        );
    }
    else if (dir== SearchRoot )
    {
        graphManager->dfv( graphManager->getMasterVertex(), getobjs );
    }
    else    // Local
    {
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            void* result = class_info.dynamicCast(*it);
            if (result != NULL && (tags.empty() || (*it)->getTags().includes(tags)))
                container(result);
        }
    }
}




void BglNode::initVisualContext()
{
    helper::vector< BglNode* > parents=getParents();
    if (parents.size())
    {

        if (showVisualModels_.getValue() == -1)
        {
            showVisualModels_.setValue(0);
            for (unsigned int i=0; i<parents.size(); ++i)
                showVisualModels_.setValue(showVisualModels_.getValue() || parents[i]->showVisualModels_.getValue());
        }
        if (showBehaviorModels_.getValue() == -1)
        {
            showBehaviorModels_.setValue(0);
            for (unsigned int i=0; i<parents.size(); ++i)
                showBehaviorModels_.setValue(showBehaviorModels_.getValue() || parents[i]->showBehaviorModels_.getValue());
        }
        if (showCollisionModels_.getValue() == -1)
        {
            showCollisionModels_.setValue(0);
            for (unsigned int i=0; i<parents.size(); ++i)
                showCollisionModels_.setValue(showCollisionModels_.getValue() || parents[i]->showCollisionModels_.getValue());
        }
        if (showBoundingCollisionModels_.getValue() == -1)
        {
            showBoundingCollisionModels_.setValue(0);
            for (unsigned int i=0; i<parents.size(); ++i)
                showBoundingCollisionModels_.setValue(showBoundingCollisionModels_.getValue() || parents[i]->showBoundingCollisionModels_.getValue());
        }
        if (showMappings_.getValue() == -1)
        {
            showMappings_.setValue(0);
            for (unsigned int i=0; i<parents.size(); ++i)
                showMappings_.setValue(showMappings_.getValue() || parents[i]->showMappings_.getValue());
        }
        if (showMechanicalMappings_.getValue() == -1)
        {
            showMechanicalMappings_.setValue(0);
            for (unsigned int i=0; i<parents.size(); ++i)
                showMechanicalMappings_.setValue(showMechanicalMappings_.getValue() || parents[i]->showMechanicalMappings_.getValue());
        }
        if (showForceFields_.getValue() == -1)
        {
            showForceFields_.setValue(0);
            for (unsigned int i=0; i<parents.size(); ++i)
                showForceFields_.setValue(showForceFields_.getValue() || parents[i]->showForceFields_.getValue());
        }
        if (showInteractionForceFields_.getValue() == -1)
        {
            showInteractionForceFields_.setValue(0);
            for (unsigned int i=0; i<parents.size(); ++i)
                showInteractionForceFields_.setValue(showInteractionForceFields_.getValue() || parents[i]->showInteractionForceFields_.getValue());
        }
        if (showWireFrame_.getValue() == -1)
        {
            showWireFrame_.setValue(0);
            for (unsigned int i=0; i<parents.size(); ++i)
                showWireFrame_.setValue(showWireFrame_.getValue() || parents[i]->showWireFrame_.getValue());
        }
        if (showNormals_.getValue() == -1)
        {
            showNormals_.setValue(0);
            for (unsigned int i=0; i<parents.size(); ++i)
                showNormals_.setValue(showNormals_.getValue() || parents[i]->showNormals_.getValue());
        }
    }
}

void BglNode::updateContext()
{
    helper::vector< BglNode* > parents=getParents();
    if (parents.size())
    {
        copyContext(*parents[0]);
    }
    simulation::Node::updateContext();
}

void BglNode::updateSimulationContext()
{
    helper::vector< BglNode* > parents=getParents();
    if (parents.size())
    {
        copySimulationContext(*parents[0]);
    }
    simulation::Node::updateSimulationContext();
}

void BglNode::updateVisualContext(int FILTER)
{
    helper::vector< BglNode* > parents=getParents();
    if (parents.size())
    {
        if (FILTER==10)
        {
            for (unsigned int i=0; i<parents.size(); ++i)
            {
                fusionVisualContext(*parents[i]);
            }
        }
        else
        {
            switch (FILTER)
            {
            case 0:
                showVisualModels_.setValue(0);
                for (unsigned int i=0; i<parents.size(); ++i)
                    showVisualModels_.setValue(showVisualModels_.getValue() || parents[i]->showVisualModels_.getValue());
                break;
            case 1:
                showBehaviorModels_.setValue(0);
                for (unsigned int i=0; i<parents.size(); ++i)
                    showBehaviorModels_.setValue(showBehaviorModels_.getValue() || parents[i]->showBehaviorModels_.getValue());
                break;
            case 2:
                showCollisionModels_.setValue(0);
                for (unsigned int i=0; i<parents.size(); ++i)
                    showCollisionModels_.setValue(showCollisionModels_.getValue() || parents[i]->showCollisionModels_.getValue());
                break;
            case 3:
                showBoundingCollisionModels_.setValue(0);
                for (unsigned int i=0; i<parents.size(); ++i)
                    showBoundingCollisionModels_.setValue(showBoundingCollisionModels_.getValue() || parents[i]->showBoundingCollisionModels_.getValue());
                break;
            case 4:
                showMappings_.setValue(0);
                for (unsigned int i=0; i<parents.size(); ++i)
                    showMappings_.setValue(showMappings_.getValue() || parents[i]->showMappings_.getValue());
                break;
            case 5:
                showMechanicalMappings_.setValue(0);
                for (unsigned int i=0; i<parents.size(); ++i)
                    showMechanicalMappings_.setValue(showMechanicalMappings_.getValue() || parents[i]->showMechanicalMappings_.getValue());
                break;
            case 6:
                showForceFields_.setValue(0);
                for (unsigned int i=0; i<parents.size(); ++i)
                    showForceFields_.setValue(showForceFields_.getValue() || parents[i]->showForceFields_.getValue());
                break;
            case 7:
                showInteractionForceFields_.setValue(0);
                for (unsigned int i=0; i<parents.size(); ++i)
                    showInteractionForceFields_.setValue(showInteractionForceFields_.getValue() || parents[i]->showInteractionForceFields_.getValue());
                break;
            case 8:
                showWireFrame_.setValue(0);
                for (unsigned int i=0; i<parents.size(); ++i)
                    showWireFrame_.setValue(showWireFrame_.getValue() || parents[i]->showWireFrame_.getValue());
                break;
            case 9:
                showNormals_.setValue(0);
                for (unsigned int i=0; i<parents.size(); ++i)
                    showNormals_.setValue(showNormals_.getValue() || parents[i]->showNormals_.getValue());
                break;
            }
        }
    }
    simulation::Node::updateVisualContext(FILTER);
}


}
}
}


