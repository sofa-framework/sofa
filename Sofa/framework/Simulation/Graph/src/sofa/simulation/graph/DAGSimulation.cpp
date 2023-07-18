/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/simulation/common/xml/BaseElement.h>
#include <sofa/simulation/graph/DAGNode.h>
#include <sofa/simulation/graph/init.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/init.h>

namespace sofa::simulation::graph
{

using namespace sofa::defaulttype;


Simulation* getSimulation()
{
    return simulation::getSimulation();
}

DAGSimulation::DAGSimulation()
{
    // Safety check; it could be elsewhere, but here is a good place, I guess.
    if (!sofa::simulation::graph::isInitialized())
        sofa::helper::printUninitializedLibraryWarning("Sofa.Simulation.Graph", "sofa::simulation::graph::init()");
}

DAGSimulation::~DAGSimulation()
{

}


Node::SPtr DAGSimulation::createNewGraph(const std::string& name)
{
    return createNewNode( name );
}

Node::SPtr DAGSimulation::createNewNode(const std::string& name)
{
    return sofa::core::objectmodel::New<DAGNode>(name);
}



// Register in the Factory
//int DAGSimulationClass = core::RegisterObject ( "Main simulation algorithm, based on tree graph" )
//.add< DAGSimulation >()
//;


} // namespace sofa::simulation::graph
