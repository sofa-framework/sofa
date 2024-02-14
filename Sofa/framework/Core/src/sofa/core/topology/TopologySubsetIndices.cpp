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
#define SOFA_CORE_TOPOLOGY_SUBSETINDICES_CPP

#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/core/topology/TopologySubsetData.inl>
#include <sofa/core/topology/TopologyDataHandler.inl>

namespace sofa::core::topology
{

TopologySubsetIndices::TopologySubsetIndices(const typename sofa::core::topology::BaseTopologyData< type::vector<Index> >::InitData& data)
    : sofa::core::topology::TopologySubsetData< core::topology::BaseMeshTopology::Point, type::vector<Index> >(data)
{

}

Index TopologySubsetIndices::indexOfElement(Index index) const
{
    const container_type& data = m_value.getValue();
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

Index TopologySubsetIndices::getLastElementIndex() const
{
    const auto nbr = Index(m_topology->getNbPoints());
    return (nbr == 0) ? sofa::InvalidID : nbr - 1;
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


void TopologySubsetIndices::addPostProcess(sofa::Index dataLastId)
{
    this->m_lastElementIndex = dataLastId;
}

void TopologySubsetIndices::updateLastIndex(Index posLastIndex, Index newGlobalId)
{
    container_type& data = *(this->beginEdit());
    data[posLastIndex] = newGlobalId;
    this->endEdit();
}

template class SOFA_CORE_API sofa::core::topology::TopologyDataHandler < core::topology::BaseMeshTopology::Point, type::vector<Index> >;
//template class SOFA_CORE_API sofa::core::topology::BaseTopologyData < type::vector<Index> >;

} //namespace sofa::core::topology
