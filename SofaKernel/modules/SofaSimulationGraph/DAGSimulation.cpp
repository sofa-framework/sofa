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
#include <SofaSimulationGraph/DAGSimulation.h>

#include <SofaSimulationCommon/xml/BaseElement.h>
#include <SofaSimulationGraph/DAGNode.h>
#include <SofaSimulationGraph/init.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/init.h>

namespace sofa
{

namespace simulation
{

namespace graph
{

using namespace sofa::defaulttype;


Simulation* getSimulation()
{
    if ( simulation::Simulation::theSimulation.get() == 0 )
    {
        setSimulation( new DAGSimulation );
    }
    return simulation::getSimulation();
}

DAGSimulation::DAGSimulation()
{
    // Safety check; it could be elsewhere, but here is a good place, I guess.
    if (!sofa::simulation::graph::isInitialized())
        sofa::helper::printUninitializedLibraryWarning("SofaSimulationGraph", "sofa::simulation::graph::init()");

    // I have no idea what this 'DuplicateEntry()' call is for, but it causes an error when we
    // create several DAGSimulation, so I added the preceding 'if' (Marc Legendre, nov. 2013)
    if (! sofa::simulation::xml::BaseElement::NodeFactory::HasKey("MultiMappingObject") )
        sofa::simulation::xml::BaseElement::NodeFactory::DuplicateEntry("DAGNodeMultiMapping","MultiMappingObject");
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



SOFA_DECL_CLASS ( DAGSimulation );
// Register in the Factory
//int DAGSimulationClass = core::RegisterObject ( "Main simulation algorithm, based on tree graph" )
//.add< DAGSimulation >()
//;


} // namespace graph

} // namespace simulation

} // namespace sofa

