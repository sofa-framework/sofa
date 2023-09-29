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
#include <sofa/core/topology/TopologyData.h>
#include <sofa/core/topology/TopologyDataHandler.inl>

namespace sofa::core::topology
{


/// static variable to be used when not all information are provided during topological event.
static const sofa::type::vector< Index > s_empty_ancestors;
static const sofa::type::vector< SReal > s_empty_coefficients;

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElementType, typename VecT>
TopologyData <ElementType, VecT>::TopologyData(const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
    : sofa::core::topology::BaseTopologyData< VecT >(data)
    , m_topologyHandler(nullptr)
    , m_isTopologyDynamic(false)
{
}

template <typename ElementType, typename VecT>
TopologyData <ElementType, VecT>::~TopologyData()
{ 
    if (m_isTopologyDynamic) {
        dmsg_info(this->getOwner()) << "TopologyData: " << this->getName() << " removed from dynamic topology: " << this->m_topology->getClassName();
        this->m_topologyHandler->unlinkFromAllTopologyDataArray();
    }
    else {
        dmsg_info(this->getOwner()) << "TopologyData: " << this->getName() << " removed from static topology without TopologyHandler.";
    }
}


template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::createTopologyHandler(sofa::core::topology::BaseMeshTopology* _topology)
{
    if (_topology == nullptr)
    {
        msg_error(this->getOwner()) << "Topology used to register this TopologyData: " << this->getName() << " is invalid. TopologyData won't be registered.";
        return;
    }
    this->m_topology = _topology;

    if (this->m_topologyHandler != nullptr)
    {
        msg_error(this->getOwner()) << "TopologyData: " << this->getName() << " already has a TopologyDataHandler. createTopologyHandler should only be called once at init of the TopologyData.";
        return;
    }

    // Create TopologyHandler
    this->m_topologyHandler = std::make_unique<TopologyDataHandler< ElementType, VecT> >(this, _topology);
    this->m_topologyHandler->setNamePrefix("TopologyDataHandler (" + this->getOwner()->getName() + ")");
    this->m_topologyHandler->init();

    // Register the TopologyHandler
    m_isTopologyDynamic = this->m_topologyHandler->registerTopology(_topology, sofa::helper::logging::notMuted(this->getOwner()));
    if (m_isTopologyDynamic)
    {
        this->linkToElementDataArray((ElementType*)nullptr);
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " initialized with dynamic " << _topology->getClassName() << "Topology.";
    }
}


/// Method used to link Data to point Data array, using the TopologyHandler's method
template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToPointDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToTopologyDataArray(sofa::geometry::ElementType::POINT);
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToPointDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToPointDataArray as toplogy is not dynamic";
}

/// Method used to link Data to edge Data array, using the TopologyHandler's method
template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToEdgeDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToTopologyDataArray(sofa::geometry::ElementType::EDGE);
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToEdgeDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToEdgeDataArray as toplogy is not dynamic";
}

/// Method used to link Data to triangle Data array, using the TopologyHandler's method
template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToTriangleDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToTopologyDataArray(sofa::geometry::ElementType::TRIANGLE);
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToTriangleDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToTriangleDataArray as toplogy is not dynamic";
}

/// Method used to link Data to quad Data array, using the TopologyHandler's method
template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToQuadDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToTopologyDataArray(sofa::geometry::ElementType::QUAD);
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToQuadDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToQuadDataArray as toplogy is not dynamic";
}

/// Method used to link Data to tetrahedron Data array, using the TopologyHandler's method
template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToTetrahedronDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToTopologyDataArray(sofa::geometry::ElementType::TETRAHEDRON);
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToTetrahedronDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToTetrahedronDataArray as toplogy is not dynamic";
}

/// Method used to link Data to hexahedron Data array, using the TopologyHandler's method
template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToHexahedronDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToTopologyDataArray(sofa::geometry::ElementType::HEXAHEDRON);
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToHexahedronDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToHexahedronDataArray as toplogy is not dynamic";
}


/////////////////////// Protected functions on TopologyData init ///////////////////////////////

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Point*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbPoints());
    linkToPointDataArray(); 
}

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Edge*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbEdges());
    linkToEdgeDataArray(); 
}

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Triangle*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbTriangles());
    linkToTriangleDataArray(); 
}

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Quad*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbQuads());
    linkToQuadDataArray(); 
}

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Tetrahedron*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbTetrahedra());
    linkToTetrahedronDataArray(); 
}

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Hexahedron*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbHexahedra());
    linkToHexahedronDataArray(); 
}


template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Point*)
{
    this->m_topologyHandler->unlinkFromTopologyDataArray(sofa::geometry::ElementType::POINT);
}

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Edge*)
{
    this->m_topologyHandler->unlinkFromTopologyDataArray(sofa::geometry::ElementType::EDGE);
}

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Triangle*)
{
    this->m_topologyHandler->unlinkFromTopologyDataArray(sofa::geometry::ElementType::TRIANGLE);
}

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Quad*)
{
    this->m_topologyHandler->unlinkFromTopologyDataArray(sofa::geometry::ElementType::QUAD);
}

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Tetrahedron*)
{
    this->m_topologyHandler->unlinkFromTopologyDataArray(sofa::geometry::ElementType::TETRAHEDRON);
}

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::unlinkFromElementDataArray(sofa::core::topology::BaseMeshTopology::Hexahedron*)
{
    this->m_topologyHandler->unlinkFromTopologyDataArray(sofa::geometry::ElementType::HEXAHEDRON);
}


///////////////////// Protected functions on TopologyData changes /////////////////////////////

template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::swap(Index i1, Index i2)
{
    helper::WriteOnlyAccessor<Data< container_type > > data = this;
    value_type tmp = data[i1];
    data[i1] = data[i2];
    data[i2] = tmp;
}


template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::remove(const sofa::type::vector<Index>& index)
{
    helper::WriteOnlyAccessor<Data< container_type > > data = this;
    if (data.size() > 0)
    {
        // make sure m_lastElementIndex is up to date before removing
        this->m_lastElementIndex = static_cast<Index>(data.size()) - 1;

        // Loop over the indices to remove. As in topology process when removing elements:
        // 1- propagate event by calling callback if it has been set.
        // 2- really remove element using swap + pop_back. 
        // 3- Update m_lastElementIndex in case it is used in callback while removing several elements
        for (std::size_t i = 0; i < index.size(); ++i)
        {
            if (p_onDestructionCallback)
            {
                p_onDestructionCallback(index[i], data[index[i]]);
            }

            this->swap(index[i], this->m_lastElementIndex);
            --this->m_lastElementIndex;
        }

        data.resize(data.size() - index.size());
    }
}


template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::add(const sofa::type::vector<Index>& index,
    const sofa::type::vector< ElementType >& elems,
    const sofa::type::vector<sofa::type::vector<Index> >& ancestors,
    const sofa::type::vector<sofa::type::vector<SReal > >& coefs,
    const sofa::type::vector< AncestorElem >& ancestorElems)
{
    SOFA_UNUSED(ancestorElems);

    std::size_t nbElements = index.size();
    if (nbElements == 0) 
        return;

    // Using default values
    helper::WriteOnlyAccessor<Data< container_type > > data = this;
    
    Index i0 = static_cast<Index>(data.size());
    if (i0 != index[0])
    {
        msg_error(this->getOwner()) << "TopologyDataHandler SIZE MISMATCH in Data "
            << this->getName() << ": " << nbElements << " "
            << geometry::ElementInfo<ElementType>::name()
            << " ADDED starting from index " << index[0]
            << " while vector size is " << i0;
        i0 = index[0];
    }


    // As in topology process when adding elements:
    // 1- Add new element. Using Data default constructors
    // 2- Update m_lastElementIndex in case it is used in callback while adding several elements
    // 3- propagate event by calling callback if it has been set.
    data.resize(i0 + nbElements);

    if (p_onCreationCallback)
    {
        for (Index i = 0; i < nbElements; ++i)
        {
            Index newElemId = i0 + i;
            value_type& t = data[newElemId];
            this->m_lastElementIndex = newElemId;

            p_onCreationCallback(newElemId, t, elems[i],
                (ancestors.empty() || coefs.empty()) ? s_empty_ancestors : ancestors[i],
                (ancestors.empty() || coefs.empty()) ? s_empty_coefficients : coefs[i]);
        }
    }
    else
    {
        this->m_lastElementIndex = static_cast<Index>(data.size()) - 1;
    }
}


template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::move(const sofa::type::vector<Index>& indexList,
    const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
    const sofa::type::vector< sofa::type::vector< SReal > >& coefs)
{
    helper::WriteOnlyAccessor<Data< container_type > > data = this;

    for (std::size_t i = 0; i < indexList.size(); i++)
    {
        if (p_onDestructionCallback)
        {
            p_onDestructionCallback(indexList[i], data[indexList[i]]);
        }

        if (p_onCreationCallback)
        {
            p_onCreationCallback(indexList[i], data[indexList[i]], ElementType(), ancestors[i], coefs[i]);
        }
    }
}



template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::renumber(const sofa::type::vector<Index>& index)
{
    helper::WriteOnlyAccessor<Data< container_type > > data = this;
    container_type copy = this->getValue(); // not very efficient memory-wise, but I can see no better solution...
    for (std::size_t i = 0; i < index.size(); ++i)
        data[i] = copy[index[i]];
}


template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::addOnMovedPosition(const sofa::type::vector<Index>& indexList,
    const sofa::type::vector<ElementType>& elems)
{
    helper::WriteOnlyAccessor<Data< container_type > > data = this;

    // Recompute data
    sofa::type::vector< Index > ancestors;
    sofa::type::vector< SReal >  coefs;
    coefs.push_back(1.0);
    ancestors.resize(1);

    if (p_onCreationCallback)
    {
        for (std::size_t i = 0; i < indexList.size(); i++)
        {
            ancestors[0] = indexList[i];
            p_onCreationCallback(indexList[i], data[indexList[i]], elems[i], ancestors, coefs);
        }
    }
    this->m_lastElementIndex += sofa::Index(indexList.size());
}


template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::removeOnMovedPosition(const sofa::type::vector<Index>& indices)
{
    helper::WriteOnlyAccessor<Data< container_type > > data = this;

    if (p_onDestructionCallback)
    {
        for (std::size_t i = 0; i < indices.size(); i++) 
        {
            p_onDestructionCallback(indices[i], data[indices[i]]);
        }       
    }

    this->m_lastElementIndex -= sofa::Index(indices.size());
}


template <typename ElementType, typename VecT>
void TopologyData <ElementType, VecT>::addTopologyEventCallBack(core::topology::TopologyChangeType type, TopologyChangeCallback callback)
{
    if (m_topologyHandler != nullptr)
    {
        m_topologyHandler->addCallBack(type, callback);
    }
    else
    {
        msg_warning(this->getOwner()) << "No TopologyHandler has been created to manage this TopologyData: " << this->getName() 
            << ". Callback for event: '" << parseTopologyChangeTypeToString(type) << "' won't be registered.";
    }

}


} //namespace sofa::core::topology
