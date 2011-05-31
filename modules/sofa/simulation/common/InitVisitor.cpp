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
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/VisualModel.h>
#include <sofa/defaulttype/BoundingBox.h>

//#include "MechanicalIntegration.h"

namespace sofa
{

namespace simulation
{


Visitor::Result InitVisitor::processNodeTopDown(simulation::Node* node)
{
    if (!rootNode) rootNode=node;

    node->initialize();
#ifdef SOFA_SMP_NUMA
    if(node->getProcessor()!=-1)
    {
        std::cerr<<"set preferred cpu "<<node->getProcessor()/2<<std::endl;
        numa_set_preferred(node->getProcessor()/2);
    }
#endif

    sofa::defaulttype::BoundingBox* nodeBBox = node->f_bbox.beginEdit(params);
    nodeBBox->invalidate();

    for(unsigned int i=0; i<node->object.size(); ++i)
    {
        node->object[i]->init();
        node->object[i]->computeBBox(params);
        nodeBBox->include(node->object[i]->f_bbox.getValue(params));
    }
    node->f_bbox.endEdit(params);
    return RESULT_CONTINUE;
}

//Move a component from the current node to the Visual Root
template <class T>
struct MoveObjectFunctor
{
    typedef  std::map< simulation::Node*, simulation::Node*> MapSimuToVisu;

    MoveObjectFunctor(MapSimuToVisu  &m):simuToVisu(m) {};

    void operator()(T *object)
    {
        Node *node=(Node*) (object->getContext());
        simulation::Node *nodeInVisualGraph;

        //we verify if we already created a node in the Visual Graph corresponding to this current Node in the Simulation Graph,
        MapSimuToVisu::iterator itFound=simuToVisu.find(node);
        if (itFound == simuToVisu.end())
        {
            //If not, we create a new node, and add it to the Visual Graph
            nodeInVisualGraph = getSimulation()->newNode(node->getName());
            getSimulation()->getVisualRoot()->addChild(nodeInVisualGraph);
            simuToVisu.insert(std::make_pair(node, nodeInVisualGraph));
            node->nodeInVisualGraph.add(nodeInVisualGraph);
            const core::objectmodel::Context &nodeContext=*static_cast<core::objectmodel::Context*>(node->getContext());
            nodeInVisualGraph->copyContext(nodeContext);
        }
        else
        {
            nodeInVisualGraph = itFound->second;
        }

        //Fill in the containers of visual components
        registerMovedComponent(node,object);
        //Effectively move the component from the node in the simulation graph to the node in the visual graph
        nodeInVisualGraph->moveObject(object);
    }

    void registerMovedComponent(Node *node, T *object)
    {
        node->componentInVisualGraph.add(object);
    }
protected:
    MapSimuToVisu &simuToVisu;
};
template <>
void MoveObjectFunctor<core::BaseMapping>::registerMovedComponent(Node *node, core::BaseMapping *object)
{
    node->visualMappingInVisualGraph.add(object);
    node->componentInVisualGraph.add(object);
}
template <>
void MoveObjectFunctor<core::VisualModel>::registerMovedComponent(Node *node, core::VisualModel *object)
{
    node->visualModelInVisualGraph.add(object);
    node->componentInVisualGraph.add(object);
}


void InitVisitor::processNodeBottomUp(simulation::Node* node)
{
    // init all the components in reverse order
    node->setDefaultVisualContextValue();
    sofa::defaulttype::BoundingBox* nodeBBox = node->f_bbox.beginEdit(params);
    Node::ChildIterator childNode;

    for( childNode = node->child.begin(); childNode!=node->child.end(); ++childNode)
    {
        nodeBBox->include((*childNode)->f_bbox.getValue(params));
    }
    node->f_bbox.endEdit(params);

    for(unsigned int i=0; i<node->object.size(); ++i)
    {
        node->object[i]->bwdInit();
    }


#ifdef SOFA_CLASSIC_SCENE_GRAPH
    return ;
#else

    using sofa::core::objectmodel::Tag;

    //Once all the children have been initialized, we move the nodes tagged as Visual
    core::objectmodel::BaseNode::Children children=node->getChildren();
    for (core::objectmodel::BaseNode::Children::iterator it=children.begin(); it!= children.end(); ++it)
    {
        core::objectmodel::BaseNode *child=(*it);
        //If the Node is tagged as Visual, we move it to the Visual Graph
        if ( child->hasTag(Tag("Visual")) )
        {
            getSimulation()->getVisualRoot()->moveChild(child);
            node->childInVisualGraph.add((Node*)child);
        }
    }


    //If we reached the Root node, we search Visual components like Visual Models and Visual Mappings to move them into the Visual Graph
    if (rootNode == node && rootNode != getSimulation()->getVisualRoot())
    {
        std::map< simulation::Node*, simulation::Node*> simuToVisu;
        if ( (Node*)(getSimulation()->getContext())  == node) //Root of the Simulation Graph
        {
            node->nodeInVisualGraph.add(getSimulation()->getVisualRoot());
            simuToVisu.insert(std::make_pair(node, getSimulation()->getVisualRoot()));
            const core::objectmodel::Context &nodeContext=*static_cast<core::objectmodel::Context*>(node->getContext());
            getSimulation()->getVisualRoot()->copyContext(nodeContext);
        }

        //********************************************************
        //Moving Visual Models
        std::list< core::VisualModel* > visualModels;
        node->getContext()->get<core::VisualModel>(&visualModels, core::objectmodel::BaseContext::SearchDown);
        MoveObjectFunctor< core::VisualModel > moveVisualModels(simuToVisu);
        std::for_each(visualModels.begin(), visualModels.end(), moveVisualModels);
        //********************************************************
        //Moving Visual Mappings
        std::list< core::BaseMapping* > mappings;
        node->getContext()->get<core::BaseMapping>(&mappings, core::objectmodel::BaseContext::SearchDown);
        //We remove from the list of mappings all the Mechanical Mappings
        std::list< core::BaseMapping* >::iterator it;
        for (it=mappings.begin(); it!=mappings.end();)
        {
            std::list< core::BaseMapping* >::iterator itMapping = it;

            ++it;

            if ((*itMapping)->isMechanical())
                mappings.erase(itMapping);
        }
        MoveObjectFunctor< core::BaseMapping > moveVisualMappings(simuToVisu);
        std::for_each(mappings.begin(), mappings.end(), moveVisualMappings);
    }
#endif

}



} // namespace simulation

} // namespace sofa

