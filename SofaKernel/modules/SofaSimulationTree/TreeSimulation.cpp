/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaSimulationTree/TreeSimulation.h>

#include <SofaSimulationCommon/xml/BaseElement.h>
#include <SofaSimulationTree/GNode.h>
#include <SofaSimulationTree/init.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/init.h>

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

TreeSimulation::TreeSimulation()
{
    // Safety check; it could be elsewhere, but here is a good place, I guess.
    if (!sofa::simulation::tree::isInitialized())
        sofa::helper::printUninitializedLibraryWarning("SofaSimulationTree", "sofa::simulation::tree::init()");

    sofa::simulation::xml::BaseElement::NodeFactory::DuplicateEntry("GNodeMultiMapping","MultiMappingObject");
}

TreeSimulation::~TreeSimulation()
{
}


Node::SPtr TreeSimulation::createNewGraph(const std::string& name)
{
    return createNewNode(name);
}

Node::SPtr TreeSimulation::createNewNode(const std::string& name)
{
    return sofa::core::objectmodel::New<GNode>(name);
}


SOFA_DECL_CLASS ( TreeSimulation )
// Register in the Factory
//int TreeSimulationClass = core::RegisterObject ( "Main simulation algorithm, based on tree graph" )
//.add< TreeSimulation >()
//;


} // namespace tree

} // namespace simulation

} // namespace sofa

