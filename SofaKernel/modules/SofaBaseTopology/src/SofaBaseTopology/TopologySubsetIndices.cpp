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
#define SOFA_COMPONENT_TOPOLOGY_SUBSETINDICES_CPP

#include <SofaBaseTopology/TopologySubsetIndices.h>
#include <SofaBaseTopology/TopologySubsetData.inl>
#include <SofaBaseTopology/TopologyDataHandler.inl>

namespace sofa::component::topology
{

TopologySubsetIndices::TopologySubsetIndices(const typename sofa::core::topology::BaseTopologyData< type::vector<Index> >::InitData& data)
    : sofa::component::topology::TopologySubsetData< core::topology::BaseMeshTopology::Point, type::vector<Index> >(data)
{

}

Index TopologySubsetIndices::indexOfElement(Index index)
{
    const container_type& data = this->getValue();
    for (Index idElem = 0; idElem < data.size(); idElem++)
    {
        if (data[idElem] == index)
            return idElem;
    }
    
    return sofa::InvalidID;
}

void TopologySubsetIndices::createTopologyHandler(sofa::core::topology::BaseMeshTopology* _topology)
{
    this->Inherit::createTopologyHandler(_topology);
}

void TopologySubsetIndices::createTopologyHandler(sofa::core::topology::BaseMeshTopology* _topology, sofa::component::topology::TopologyDataHandler < core::topology::BaseMeshTopology::Point, type::vector<Index> >* topoEngine)
{
    this->Inherit::createTopologyHandler(_topology, topoEngine);
}


void TopologySubsetIndices::swapPostProcess(Index i1, Index i2)
{
    // nothing to do here
    SOFA_UNUSED(i1);
    SOFA_UNUSED(i2);
}


void TopologySubsetIndices::removePostProcess(sofa::Size nbElements)
{
    // nothing to do here
    SOFA_UNUSED(nbElements);
}


void TopologySubsetIndices::addPostProcess(sofa::Size nbElements)
{
    this->m_lastElementIndex += nbElements;
}

void TopologySubsetIndices::updateLastIndex(Index posLastIndex, Index newGlobalId)
{
    container_type& data = *(this->beginEdit());
    data[posLastIndex] = newGlobalId;
    this->endEdit();
}

template class SOFA_SOFABASETOPOLOGY_API sofa::component::topology::TopologyDataHandler < core::topology::BaseMeshTopology::Point, type::vector<Index> >;
template class SOFA_SOFABASETOPOLOGY_API sofa::component::topology::TopologyData < core::topology::BaseMeshTopology::Point, type::vector<Index> >;
//template class SOFA_SOFABASETOPOLOGY_API sofa::core::topology::BaseTopologyData < type::vector<Index> >;

} //namespace sofa::component::topology
