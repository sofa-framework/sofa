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
#include <sofa/core/topology/TopologySubsetData.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/core/topology/TopologyDataHandler.inl>

namespace sofa::core::topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TopologyElementType, typename VecT>
TopologySubsetData <TopologyElementType, VecT>::TopologySubsetData(const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
    : sofa::core::topology::TopologyData< TopologyElementType, VecT >(data)
{

}

///////////////////// Private functions on TopologySubsetData changes /////////////////////////////
template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::swap(Index i1, Index i2)
{
    container_type& data = *(this->beginEdit());
    
    if (i1 >= data.size() || i2 >= data.size())
    {
        msg_warning("TopologySubsetData") << "swap indices out of bouds: i1: " << i1 << " | i2: " << i2 << " out of data size: " << data.size();
        this->endEdit();
        return;
    }

    value_type tmp = data[i1];
    data[i1] = data[i2];
    data[i2] = tmp;

    this->endEdit();

    swapPostProcess(i1, i2);
}

template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::setMap2Elements(const sofa::type::vector<Index> _map2Elements)
{
    m_map2Elements = _map2Elements;
}

template <typename TopologyElementType, typename VecT>
Index TopologySubsetData <TopologyElementType, VecT>::indexOfElement(Index index)
{    
    for (unsigned int i = 0; i < m_map2Elements.size(); ++i)
        if (index == m_map2Elements[i])
            return i;

    return sofa::InvalidID;
}

template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::add(sofa::Size nbElements,
    const sofa::type::vector<sofa::type::vector<Index> >& ancestors,
    const sofa::type::vector<sofa::type::vector<SReal> >& coefs)
{
    sofa::type::vector< TopologyElementType > elems; 
    elems.resize(nbElements);

    return this->add(nbElements, elems, ancestors, coefs);    
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::add(sofa::Size nbElements,
    const sofa::type::vector< TopologyElementType >& elems,
    const sofa::type::vector<sofa::type::vector<Index> >& ancestors,
    const sofa::type::vector<sofa::type::vector<SReal> >& coefs)
{
    // Track TopologyData last index before applying changes. special case if id is invalid == start with empty buffer
    int LastDataId = (this->m_lastElementIndex == sofa::InvalidID) ? -1 : int(this->m_lastElementIndex);

    // if no new element are added to this subset. Just update the lastElementIndex for future deletion
    if (!this->isNewTopologyElementsSupported())
    {
        this->m_lastElementIndex = Index(LastDataId + nbElements);
        return;
    }

    helper::WriteOnlyAccessor<Data<container_type> > data = this;

    // first resize the subsetData. value will be applied in the loop using callbacks
    Size size = data.size();
    data.resize(size + nbElements);
    

    if (this->p_onCreationCallback)
    {
        for (std::size_t i = 0; i < nbElements; ++i)
        {
            Index id = Index(size + i);
            value_type& t = data[id];
            
            // update map if needed
            addPostProcess(LastDataId + i + 1);

            this->p_onCreationCallback(id, t, elems[i], 
                (ancestors.empty() || coefs.empty()) ? s_empty_ancestors : ancestors[i],
                (ancestors.empty() || coefs.empty()) ? s_empty_coefficients : coefs[i]);

        }
    }
    else
    {
        this->m_lastElementIndex = Index(LastDataId + nbElements);
    }

}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::add(const sofa::type::vector<Index>& index,
    const sofa::type::vector< TopologyElementType >& elems,
    const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
    const sofa::type::vector< sofa::type::vector< SReal > >& coefs,
    const sofa::type::vector< AncestorElem >& ancestorElems)
{
    SOFA_UNUSED(ancestorElems);

    const sofa::Size nbElements = index.size();
    this->add(nbElements, elems, ancestors, coefs);
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::move(const sofa::type::vector<Index>&,
    const sofa::type::vector< sofa::type::vector< Index > >&,
    const sofa::type::vector< sofa::type::vector< SReal > >&)
{

}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::remove(const sofa::type::vector<Index>& index)
{
    helper::WriteOnlyAccessor<Data<container_type> > data = this;
    
    // Update last element index before removing elements. Warn is sent before updating Topology buffer
    Index lastTopoElemId = this->getLastElementIndex();
    
    // check for each element index to remove if it concern this subsetData
    for (Index elemId : index)
    {
        // Check if this element is inside the subset map
        Index dataId = this->indexOfElement(elemId);
        
        if (dataId != sofa::InvalidID) // index in the map, need to update the subsetData
        {
            // if in the map, apply callback if set
            if (this->p_onDestructionCallback)
            {
                this->p_onDestructionCallback(dataId, data[dataId]);
            }

            // Like in topological change, will swap before poping back
            Index lastDataId = data.size() - 1;
            this->swap(dataId, lastDataId);

            // Remove last subsetData element and update the map
            data.resize(lastDataId);
            removePostProcess(lastDataId);
        }

        // Need to check if last element index is in the map. If yes need to replace that value to follow topological changes
        if (lastTopoElemId == sofa::InvalidID)
            continue;

        dataId = this->indexOfElement(lastTopoElemId);
        if (dataId != sofa::InvalidID)
        {
            updateLastIndex(dataId, elemId);
        }
        lastTopoElemId--;
    }
}

template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::renumber(const sofa::type::vector<Index>& index)
{
    container_type& data = *(this->beginEdit());
    container_type copy = this->getValue(); // not very efficient memory-wise, but I can see no better solution...

    for (std::size_t i = 0; i < data.size(); ++i)
    {
        data[i] = copy[index[i]];
    }
    this->endEdit();
}



template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::swapPostProcess(Index i1, Index i2)
{
    if (i1 >= m_map2Elements.size() || i2 >= m_map2Elements.size())
    {
        msg_warning("TopologySubsetData") << "swap indices out of bouds: i1: " << i1 << " | i2: " << i2 << " out of m_map2Elements size: " << m_map2Elements.size();
        return;
    }

    //apply same change to map:
    Index tmp2 = m_map2Elements[i1];
    m_map2Elements[i1] = m_map2Elements[i2];
    m_map2Elements[i2] = tmp2;
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::removePostProcess(sofa::Size nbElements)
{
    m_map2Elements.resize(nbElements);
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData<TopologyElementType, VecT>::addPostProcess(sofa::Index dataLastId)
{
    this->m_lastElementIndex = dataLastId;
    m_map2Elements.push_back(dataLastId);
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::updateLastIndex(Index posLastIndex, Index newGlobalId)
{
    m_map2Elements[posLastIndex] = newGlobalId;
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::addOnMovedPosition(const sofa::type::vector<Index>&,
    const sofa::type::vector<TopologyElementType>&)
{
    dmsg_error("TopologySubsetData") << "addOnMovedPosition event on topology subsetData is not yet handled.";
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::removeOnMovedPosition(const sofa::type::vector<Index>&)
{
    dmsg_error("TopologySubsetData") << "removeOnMovedPosition event on topology subsetData is not yet handled";
}


} //namespace sofa::core::topology
