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
#include <sofa/core/componentmodel/behavior/BaseMechanicalMapping.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/VisualModel.h>

//#include "MechanicalIntegration.h"

namespace sofa
{

namespace simulation
{


Visitor::Result InitVisitor::processNodeTopDown(simulation::Node* node)
{
    if (!rootNode) rootNode=node;

    node->initialize();


    for(unsigned int i=0; i<node->object.size(); ++i)
    {
        node->object[i]->init();
        if (node->object[i]->canPrefetch())
        {
            simulation::getSimulation()->setPrefetching( node->object[i]->canPrefetch() );
        }
    }
    return RESULT_CONTINUE;
}

template <class T>
struct MoveObjectFunctor
{
    void operator()(T *object)
    {
        getSimulation()->getVisualRoot()->moveObject(object);
    }
};


void InitVisitor::processNodeBottomUp(simulation::Node* node)
{
    // init all the components in reverse order
    node->setDefaultVisualContextValue();

    for(unsigned int i=0; i<node->object.size(); ++i)
    {
        node->object[i]->bwdInit();
    }

    using sofa::core::objectmodel::Tag;

    //Once all the children have been initialized, we move the nodes tagged as Visual
    core::objectmodel::BaseNode::Children children=node->getChildren();
    for (core::objectmodel::BaseNode::Children::iterator it=children.begin(); it!= children.end(); ++it)
    {
        core::objectmodel::BaseNode *node=(*it);
        //If the Node is tagged as Visual, we move it to the Visual Graph
        if ( node->hasTag(Tag("Visual")) )
        {
            getSimulation()->getVisualRoot()->moveChild(node);
        }
    }

    //If we reached the Root node, we search Visual components like Visual Models and Visual Mappings to move them into the Visual Graph
    if (rootNode == node && rootNode != getSimulation()->getVisualRoot())
    {
        //********************************************************
        //Moving Visual Models
        std::list< core::VisualModel* > visualModels;
        node->getContext()->get<core::VisualModel>(&visualModels, core::objectmodel::BaseContext::SearchDown);
        MoveObjectFunctor< core::VisualModel > moveVisualModels;
        std::for_each(visualModels.begin(), visualModels.end(), moveVisualModels);
        //********************************************************
        //Moving Visual Mappings
        std::list< core::BaseMapping* > mappings;
        node->getContext()->get<core::BaseMapping>(&mappings, core::objectmodel::BaseContext::SearchDown);
        //We remove from the list of mappings all the Mechanical Mappings
        std::list< core::BaseMapping* >::iterator it;
        for (it=mappings.begin(); it!=mappings.end();)
        {
            std::list< core::BaseMapping* >::iterator itMapping=it;
            ++it;
            if (core::componentmodel::behavior::BaseMechanicalMapping* m=dynamic_cast<core::componentmodel::behavior::BaseMechanicalMapping*>( *itMapping ))
            {
                if (m->isMechanical())
                    mappings.erase(itMapping);
            }
        }
        MoveObjectFunctor< core::BaseMapping > moveVisualMappings;
        std::for_each(mappings.begin(), mappings.end(), moveVisualMappings);
    }

}



} // namespace simulation

} // namespace sofa

