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
#define SOFA_CORE_TOPOLOGY_TOPOLOGYENGINE_DEFINITION true
#include <sofa/core/topology/TopologyEngine.h>
#include <sofa/core/topology/TopologyChange.h>

#ifdef SOFA_CORE_TOPOLOGY_TOPOLOGYENGINE_DEFINITION
namespace std
{
    template class list<const sofa::core::topology::TopologyChange*>;
}

namespace sofa::core::objectmodel
{
template class Data<std::list<const sofa::core::topology::TopologyChange*>>;
}
#endif /// SOFA_CORE_TOPOLOGY_TOPOLOGYENGINE_DEFINITION


namespace sofa::core::topology
{

void TopologyEngine::init()
{
    sofa::core::DataEngine::init();
    this->createEngineName();
}

size_t TopologyEngine::getNumberOfTopologicalChanges()
{
    return (m_changeList.getValue()).size();
}

void TopologyEngine::createEngineName()
{
    if (m_data_name.empty())
        setName( m_prefix + "no_name" );
    else
        setName( m_prefix + m_data_name );

    return;
}


} // namespace sofa

