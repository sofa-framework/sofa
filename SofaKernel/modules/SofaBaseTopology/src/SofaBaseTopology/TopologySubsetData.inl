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
#pragma once
#include <SofaBaseTopology/TopologySubsetData.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseTopology/TopologyDataEngine.inl>
#include <SofaBaseTopology/TopologySubsetDataHandler.inl>

namespace sofa::component::topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology, sofa::core::topology::TopologyHandler *_topologyHandler, bool /*deleteHandler*/)
{
    this->m_topology = _topology;
    if (_topology && dynamic_cast<sofa::core::topology::TopologyContainer*>(_topology))
    {
        this->m_topologicalEngine = sofa::core::objectmodel::New< TopologyDataEngine<VecT> >((sofa::component::topology::TopologySubsetData<TopologyElementType, VecT>*)this, _topology, _topologyHandler);
        this->m_topologicalEngine->setNamePrefix(std::string(sofa::core::topology::TopologyElementInfo<TopologyElementType>::name()) + std::string("SubsetEngine_"));
        if (this->getOwner() && dynamic_cast<sofa::core::objectmodel::BaseObject*>(this->getOwner())) dynamic_cast<sofa::core::objectmodel::BaseObject*>(this->getOwner())->addSlave(this->m_topologicalEngine.get());
        this->m_topologicalEngine->init();
        this->linkToElementDataArray((TopologyElementType*)nullptr);
        msg_info(this->getOwner())<<"TopologySubsetData: " << this->getName() << " initialized with dynamic " << _topology->getClassName() << " Topology.";
    }
    else if (_topology)
    {
        msg_info(this->getOwner())<<"TopologySubsetData: " << this->getName() << " initialized with static " << _topology->getClassName() << " Topology.";
    }
    else
    {
        msg_info(this->getOwner())<<"TopologySubsetData: No Topology given to " << this->getName() << " to createTopologicalEngine. Topological changes will be disabled.";
    }
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::createTopologicalEngine(sofa::core::topology::BaseMeshTopology *_topology)
{
    this->m_topologyHandler = new TopologySubsetDataHandler<TopologyElementType, VecT>(this);
    createTopologicalEngine(_topology, this->m_topologyHandler);
}

} //namespace sofa::component::topology
