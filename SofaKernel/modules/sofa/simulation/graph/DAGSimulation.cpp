/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/simulation/common/xml/BaseElement.h>
#include <sofa/simulation/graph/DAGNode.h>
#include <sofa/simulation/graph/init.h>

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
    unload(sRoot);
    sRoot = sofa::core::objectmodel::New<DAGNode>(name);
    return sRoot;
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

