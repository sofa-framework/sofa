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
#include "BglSimulation.h"
#include "GetObjectsVisitor.h"


#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/componentmodel/behavior/BaseMechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/InteractionForceField.h>
#include <sofa/core/componentmodel/behavior/InteractionConstraint.h>


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

BglNode::BglNode(BglSimulation* s,const std::string& name)
    : sofa::simulation::Node(name), scene(s), graph(NULL)
{

}

BglNode::BglNode(BglSimulation* s, BglSimulation::Hgraph *g,  BglSimulation::Hvertex n, const std::string& name)
    : sofa::simulation::Node(name), scene(s), graph(g), vertexId(n)
{

}


BglNode::~BglNode()
{
}

bool BglNode::addObject(BaseObject* obj)
{
    if (sofa::core::componentmodel::behavior::BaseMechanicalMapping* mm = dynamic_cast<sofa::core::componentmodel::behavior::BaseMechanicalMapping*>(obj))
    {
        scene->setMechanicalMapping(this,mm);
        return true;
    }
    else if (sofa::core::componentmodel::behavior::InteractionForceField* iff = dynamic_cast<sofa::core::componentmodel::behavior::InteractionForceField*>(obj))
    {
        scene->setContactResponse(this,iff);
        return true;
    }
    else if (sofa::core::componentmodel::behavior::InteractionConstraint* ic = dynamic_cast<sofa::core::componentmodel::behavior::InteractionConstraint*>(obj))
    {
        scene->setContactResponse(this,ic);
        return true;
    }
    return Node::addObject(obj);
}

bool BglNode::removeObject(core::objectmodel::BaseObject* obj)
{
    if (sofa::core::componentmodel::behavior::BaseMechanicalMapping* mm = dynamic_cast<sofa::core::componentmodel::behavior::BaseMechanicalMapping*>(obj))
    {
        scene->resetMechanicalMapping(this,mm);
        return true;
    }
    else if (sofa::core::componentmodel::behavior::InteractionForceField* iff = dynamic_cast<sofa::core::componentmodel::behavior::InteractionForceField*>(obj))
    {
        scene->resetContactResponse(this,iff);
        return true;
    }
    else if (sofa::core::componentmodel::behavior::InteractionConstraint* ic = dynamic_cast<sofa::core::componentmodel::behavior::InteractionConstraint*>(obj))
    {
        scene->resetContactResponse(this,ic);
        return true;
    }
    return Node::removeObject(obj);
}

void BglNode::addChild(Node* c)
{
//         std::cerr << "addChild : of "<< this->getName() << "@" << this << " and " <<  c->getName() << "@" << c << "\n";
    scene->addNode(this,dynamic_cast<BglNode*>(c));
}

void BglNode::removeChild(Node* c)
{
//         std::cerr << "deleteChild : of "<< this->getName() << "@" << this << " and " <<  c->getName() << "@" << c << "\n";
    scene->deleteNode(c);
}

void BglNode::moveChild(Node* c)
{
    std::cerr << "moveChild : " << c << "\n";
    //We have to remove all the in-edges pointing to the node "c", and add c as a child of this current node
    if (BglNode *bglNode = dynamic_cast< BglNode* >(c))
    {
        BglSimulation::Hvertex childId =bglNode->getVertexId();
        BglSimulation::Rvertex childIdR=scene->r_node_vertex_map[bglNode];
        BglSimulation::Hgraph::in_edge_iterator in_i, in_end;
        BglSimulation::Hgraph &g=bglNode->getGraph();
        //Find all in-edges from the graph
        for (tie(in_i, in_end) = boost::in_edges(childId, g); in_i != in_end; ++in_i)
        {
            BglSimulation::Hedge e=*in_i;
            BglSimulation::Hvertex src=source(e, g);
            //Remove previous edges
            boost::clear_vertex(childId, scene->hgraph);
            boost::remove_vertex(childId, scene->hgraph);

            boost::clear_vertex(childIdR, scene->rgraph);
            boost::remove_vertex(childIdR, scene->rgraph);
        }

        scene->addHedge(vertexId, childId);
    }
}


/// Find all the Nodes pointing
helper::vector< BglNode* > BglNode::getParents()
{
    helper::vector< BglNode* > p;
    if (!graph) return p;
    BglSimulation::Hgraph::in_edge_iterator in_i, in_end;
    //Find all in-edges from the graph
    for (tie(in_i, in_end) = boost::in_edges(vertexId, *graph); in_i != in_end; ++in_i)
    {
        BglSimulation::Hedge e=*in_i;
        BglSimulation::Hvertex src=source(e, *graph);
        p.push_back(static_cast<BglNode*>(scene->h_vertex_node_map[src]));
    }
    return p;
}


void BglNode::doExecuteVisitor( Visitor* vis )
{
    //cerr<<"BglNode::doExecuteVisitor( simulation::tree::Visitor* action)"<<endl;

    boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(*graph) );
    //boost::queue<BglSimulation::Hvertex> queue;

    /*    boost::breadth_first_search(
          graph,
          boost::vertex(this->vertexId, *graph),
          queue,
          bfs_adapter(vis,scene->h_vertex_node_map),
          colors
          );*/

    dfv_adapter dfv(vis,scene->h_vertex_node_map);
    boost::depth_first_visit(
        *graph,
        boost::vertex(this->vertexId, *graph),
        dfv,
        colors,
        dfv
    );
}

void BglNode::clearInteractionForceFields()
{
    for (unsigned int i=0; i<interactionForceField.size(); ++i)
        scene->removeInteraction(interactionForceField[i]);
    interactionForceField.clear();
}



/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BglNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, SearchDirection dir) const
{
    GetObjectVisitor getobj(class_info);
    if ( dir == SearchDown )
    {
//             std::cerr << "Search Down ";
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(scene->hgraph) );
        dfv_adapter dfv( &getobj, scene->h_vertex_node_map );
        boost::depth_first_visit(
            scene->hgraph,
            boost::vertex(this->vertexId, scene->hgraph),
            dfv,
            colors,
            dfv
        );
    }
    else if (dir== SearchUp )
    {
//             std::cerr << "Search Up ";
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(scene->rgraph) );
        dfv_adapter dfv( &getobj, scene->r_vertex_node_map );
        BglSimulation::Rvertex thisvertex = scene->r_node_vertex_map[scene->h_vertex_node_map[this->vertexId]];
        boost::depth_first_visit(
            scene->rgraph,
            boost::vertex(thisvertex, scene->rgraph),
            dfv,
            colors,
            dfv
        );
    }
    else if (dir== SearchRoot )
    {
//             std::cerr << "Search Root ";
        scene->dfv( scene->masterVertex, getobj );
    }
    else    // Local
    {
//             std::cerr << "Search Local ";
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            void* result = class_info.dynamicCast(*it);
            if (result != NULL)
            {
//                     std::cerr << "Single Search : " << sofa::helper::gettypename((class_info)) << " result : " << result << "\n";
                return result;
            }
        }
    }
//         std::cerr << "Single Search : " << sofa::helper::gettypename(class_info) << " result : " << getobj.getObject() << "\n";
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
void BglNode::getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir) const
{
//         std::cerr << "Search for " << sofa::helper::gettypename(class_info) << "\n";
    GetObjectsVisitor getobjs(class_info, container);
    if ( dir == SearchDown )
    {
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(scene->hgraph) );
        dfv_adapter dfv( &getobjs, scene->h_vertex_node_map );
        boost::depth_first_visit(
            scene->hgraph,
            boost::vertex(this->vertexId, scene->hgraph),
            dfv,
            colors,
            dfv
        );
    }
    else if (dir== SearchUp )
    {
        boost::vector_property_map<boost::default_color_type> colors( boost::num_vertices(scene->rgraph) );
        dfv_adapter dfv( &getobjs, scene->r_vertex_node_map );
        BglSimulation::Rvertex thisvertex = scene->r_node_vertex_map[scene->h_vertex_node_map[this->vertexId]];
        boost::depth_first_visit(
            scene->rgraph,
            boost::vertex(thisvertex, scene->rgraph),
            dfv,
            colors,
            dfv
        );
    }
    else if (dir== SearchRoot )
    {
        scene->dfv( scene->masterVertex, getobjs );
    }
    else    // Local
    {
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            void* result = class_info.dynamicCast(*it);
            if (result != NULL)
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
                fusionVisualContext(*parents[i]);
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


