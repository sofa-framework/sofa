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
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/simulation/tree/GNode.h>

#include <sofa/simulation/common/xml/BaseElement.h>
#include <sofa/helper/Factory.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

using namespace sofa::defaulttype;


Simulation* getSimulation()
{
    if ( simulation::Simulation::theSimulation.get() == 0 )
    {
        setSimulation( new TreeSimulation );
    }
    return simulation::getSimulation();
}

TreeSimulation::TreeSimulation(): visualNode(NULL)
{
    //-------------------------------------------------------------------------------------------------------
    sofa::core::ObjectFactory::AddAlias("DefaultCollisionGroupManager",
            "TreeCollisionGroupManager", true, 0);

    sofa::core::ObjectFactory::AddAlias("CollisionGroup",
            "TreeCollisionGroupManager", true, 0);



    sofa::simulation::xml::BaseElement::NodeFactory::DuplicateEntry("GNodeMultiMapping","MultiMappingObject");
}

TreeSimulation::~TreeSimulation()
{

}

Node *TreeSimulation::getVisualRoot()
{
    if (visualNode.get() == 0)
    {
        visualNode.reset(new GNode("VisualNode"));
        visualNode->addTag(core::objectmodel::Tag("Visual"));
    }
    return visualNode.get();
}

/// Create a new graph
Node* TreeSimulation::createNewGraph(const std::string& name)
{
    return new GNode(name);
}

/// Create a new node
Node* TreeSimulation::newNode(const std::string& name)
{
    return new GNode(name);
}

SOFA_DECL_CLASS ( TreeSimulation );
// Register in the Factory
int TreeSimulationClass = core::RegisterObject ( "Main simulation algorithm, based on tree graph" )
        .add< TreeSimulation >()
        ;


} // namespace tree

} // namespace simulation

} // namespace sofa

