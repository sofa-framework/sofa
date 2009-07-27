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

//Components of the core to detect during the addition of objects in a node
#include <sofa/core/componentmodel/behavior/InteractionForceField.h>
#include <sofa/core/componentmodel/behavior/InteractionConstraint.h>


#include "dfv_adapter.h"
#include <boost/vector_property_map.hpp>


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


bool BglNode::addObject(BaseObject* obj)
{
    if (sofa::core::componentmodel::behavior::BaseMechanicalMapping* mm = dynamic_cast<sofa::core::componentmodel::behavior::BaseMechanicalMapping*>(obj))
    {
        if (mm->getMechFrom() == NULL)
        {
            std::cerr << "ERROR in addObject BglNode: RayPick Issue!!\n"; return false;
        }
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
    else if (// sofa::core::componentmodel::behavior::MasterSolver* ms =
        dynamic_cast<sofa::core::componentmodel::behavior::MasterSolver*>(obj))
    {
        graphManager->addSolver(this);
    }
    else if (// sofa::core::componentmodel::behavior::OdeSolver* odes =
        dynamic_cast<sofa::core::componentmodel::behavior::OdeSolver*>(obj))
    {
        graphManager->addSolver(this);
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


/// Add a child node
void BglNode::doAddChild(BglNode* node)
{
    child.add(node);
    node->parents.add(this);
    graphManager->addNode(this,node);
}


void BglNode::addChild(core::objectmodel::BaseNode* c)
{
    BglNode *childNode = static_cast< BglNode *>(c);
    //std::cerr << "addChild : of "<< this->getName() << "@" << this << " and " <<  c->getName() << "@" << c << "\n";

    notifyAddChild(childNode);
    doAddChild(childNode);
}

/// Remove a child
void BglNode::doRemoveChild(BglNode* node)
{
    child.remove(node);
    node->parents.remove(this);
    graphManager->deleteNode(node);
}

void BglNode::removeChild(core::objectmodel::BaseNode* c)
{
    BglNode *childNode = static_cast< BglNode *>(c);
    //std::cerr << "deleteChild : of "<< this->getName() << "@" << this << " and " <<  c->getName() << "@" << c << "\n";

    notifyRemoveChild(childNode);
    doRemoveChild(childNode);
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
            prev->doRemoveChild(childNode);
        }
        doAddChild(childNode);
    }
}

void BglNode::detachFromGraph()
{
    for (ParentIterator it=parents.begin(); it!=parents.end(); ++it) (*it)->removeChild(this);
}


helper::vector< const BglNode* > BglNode::getParents() const
{
    helper::vector< const BglNode* > p;
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


helper::vector< BglNode* > BglNode::getParents()
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
    if (!parents.empty()) str = (*parents.begin())->getPathName();
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





void BglNode::doExecuteVisitor( Visitor* vis )
{
    //cerr<<"BglNode::doExecuteVisitor( simulation::tree::Visitor* action)"<<endl;
    if (!graph) return;

    boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(*graph) );

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
    if (graphManager->isNodeCreated(this))
    {
        std::cerr << "ERROR : getObject(" << sofa::helper::gettypename(class_info) << "," << dir << ") not done!!!!! Node " << this->getName() << " still not initialized\n";
        return NULL;
    }
    GetObjectVisitor getobj(class_info);
    getobj.setTags(tags);
    if ( dir == SearchDown )
    {
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
        graphManager->dfv( graphManager->getMasterVertex(), getobj );
    }
    else    // Local
    {
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            void* result = class_info.dynamicCast(*it);
            if (result != NULL && (tags.empty() || (*it)->getTags().includes(tags)))
            {
                return result;
            }
        }
    }
    return getobj.getObject();
}


/// Generic object access, given a path from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BglNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const
{
    if (path.empty())
    {
        return Node::getObject(class_info, Local);
    }
    else if (path[0] == '/')
    {
        if (!parents.empty())
        {
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
            {
                void *result=(*it)->getObject(class_info, path);
                if (result) return result;
            }
            return NULL;
        }
        else return getObject(class_info,std::string(path,1));
    }
    else if (std::string(path,0,2)==std::string("./"))
    {
        std::string newpath = std::string(path, 2);
        while (!newpath.empty() && path[0] == '/')
            newpath.erase(0);
        return getObject(class_info,newpath);
    }
    else if (std::string(path,0,3)==std::string("../"))
    {
        std::string newpath = std::string(path, 3);
        while (!newpath.empty() && path[0] == '/')
            newpath.erase(0);

        if (!parents.empty())
        {
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
            {
                void *result=(*it)->getObject(class_info, newpath);
                if (result) return result;
            }
            return NULL;
        }
        else return getObject(class_info,newpath);
    }
    else
    {
        std::string::size_type pend = path.find('/');
        if (pend == std::string::npos) pend = path.length();
        std::string name ( path, 0, pend );
        Node* child = getChild(name);
        if (child)
        {
            while (pend < path.length() && path[pend] == '/')
                ++pend;
            return child->getObject(class_info, std::string(path, pend));
        }
        else if (pend < path.length())
        {
            std::cerr << "ERROR: child node "<<name<<" not found in "<<getPathName()<<std::endl;
            return NULL;
        }
        else
        {
            core::objectmodel::BaseObject* obj = simulation::Node::getObject(name);
            if (obj == NULL)
            {
                std::cerr << "ERROR: object "<<name<<" not found in "<<getPathName()<<std::endl;
                return NULL;
            }
            else
            {
                void* result = class_info.dynamicCast(obj);
                if (result == NULL)
                {
                    std::cerr << "ERROR: object "<<name<<" in "<<getPathName()<<" does not implement class "<<class_info.name()<<std::endl;
                    return NULL;
                }
                else
                {
                    return result;
                }
            }
        }
    }

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
    if (!parents.empty())
    {
        if (showVisualModels_.getValue() == -1)
        {
            showVisualModels_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showVisualModels_.setValue(showVisualModels_.getValue() || (*it)->showVisualModels_.getValue());
        }
        if (showBehaviorModels_.getValue() == -1)
        {
            showBehaviorModels_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showBehaviorModels_.setValue(showBehaviorModels_.getValue() || (*it)->showBehaviorModels_.getValue());
        }
        if (showCollisionModels_.getValue() == -1)
        {
            showCollisionModels_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showCollisionModels_.setValue(showCollisionModels_.getValue() || (*it)->showCollisionModels_.getValue());
        }
        if (showBoundingCollisionModels_.getValue() == -1)
        {
            showBoundingCollisionModels_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showBoundingCollisionModels_.setValue(showBoundingCollisionModels_.getValue() || (*it)->showBoundingCollisionModels_.getValue());
        }
        if (showMappings_.getValue() == -1)
        {
            showMappings_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showMappings_.setValue(showMappings_.getValue() || (*it)->showMappings_.getValue());
        }
        if (showMechanicalMappings_.getValue() == -1)
        {
            showMechanicalMappings_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showMechanicalMappings_.setValue(showMechanicalMappings_.getValue() || (*it)->showMechanicalMappings_.getValue());
        }
        if (showForceFields_.getValue() == -1)
        {
            showForceFields_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showForceFields_.setValue(showForceFields_.getValue() || (*it)->showForceFields_.getValue());
        }
        if (showInteractionForceFields_.getValue() == -1)
        {
            showInteractionForceFields_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showInteractionForceFields_.setValue(showInteractionForceFields_.getValue() || (*it)->showInteractionForceFields_.getValue());
        }
        if (showWireFrame_.getValue() == -1)
        {
            showWireFrame_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showWireFrame_.setValue(showWireFrame_.getValue() || (*it)->showWireFrame_.getValue());
        }
        if (showNormals_.getValue() == -1)
        {
            showNormals_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showNormals_.setValue(showNormals_.getValue() || (*it)->showNormals_.getValue());
        }
    }
}

void BglNode::updateContext()
{
    if (!parents.empty())
    {
        copyContext(*(*parents.begin()));
    }
    simulation::Node::updateContext();
}

void BglNode::updateSimulationContext()
{
    if (!parents.empty())
    {
        copySimulationContext(*(*parents.begin()));
    }
    simulation::Node::updateSimulationContext();
}

void BglNode::updateVisualContext(VISUAL_FLAG FILTER)
{
    if (!parents.empty())
    {
        switch (FILTER)
        {
        case VISUALMODELS:
            showVisualModels_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showVisualModels_.setValue(showVisualModels_.getValue() || (*it)->showVisualModels_.getValue());
            break;
        case BEHAVIORMODELS:
            showBehaviorModels_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showBehaviorModels_.setValue(showBehaviorModels_.getValue() || (*it)->showBehaviorModels_.getValue());
            break;
        case COLLISIONMODELS:
            showCollisionModels_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showCollisionModels_.setValue(showCollisionModels_.getValue() || (*it)->showCollisionModels_.getValue());
            break;
        case BOUNDINGCOLLISIONMODELS:
            showBoundingCollisionModels_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showBoundingCollisionModels_.setValue(showBoundingCollisionModels_.getValue() || (*it)->showBoundingCollisionModels_.getValue());
            break;
        case MAPPINGS:
            showMappings_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showMappings_.setValue(showMappings_.getValue() || (*it)->showMappings_.getValue());
            break;
        case MECHANICALMAPPINGS:
            showMechanicalMappings_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showMechanicalMappings_.setValue(showMechanicalMappings_.getValue() || (*it)->showMechanicalMappings_.getValue());
            break;
        case FORCEFIELDS:
            showForceFields_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showForceFields_.setValue(showForceFields_.getValue() || (*it)->showForceFields_.getValue());
            break;
        case INTERACTIONFORCEFIELDS:
            showInteractionForceFields_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showInteractionForceFields_.setValue(showInteractionForceFields_.getValue() || (*it)->showInteractionForceFields_.getValue());
            break;
        case WIREFRAME:
            showWireFrame_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showWireFrame_.setValue(showWireFrame_.getValue() || (*it)->showWireFrame_.getValue());
            break;
        case NORMALS:
            showNormals_.setValue(0);
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
                showNormals_.setValue(showNormals_.getValue() || (*it)->showNormals_.getValue());
            break;
        case ALLFLAGS:
            for (ParentIterator it=parents.begin(); it!=parents.end(); ++it)
            {
                fusionVisualContext(*(*it));
            }
            break;
        }
    }
    simulation::Node::updateVisualContext(FILTER);
}

SOFA_DECL_CLASS(BglNode)

helper::Creator<simulation::tree::xml::NodeElement::Factory, BglNode> BglNodeClass("BglNode");

}
}
}


