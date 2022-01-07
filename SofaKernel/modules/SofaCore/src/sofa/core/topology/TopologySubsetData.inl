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
    , m_isConcerned(false)
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
    if (!this->getSparseDataStatus()) {
        this->m_lastElementIndex += nbElements;
        return;
    }

    // Using default values
    container_type& data = *(this->beginEdit());

    Size size = data.size();
    data.resize(size + nbElements);
    
    // Call for specific callback if handler has been set
    if (this->m_topologyHandler)
    {
        value_type t;
        for (std::size_t i = 0; i < nbElements; ++i)
        {
            if (ancestors.empty() || coefs.empty())
            {
                const sofa::type::vector< Index > empty_vecint;
                const sofa::type::vector< SReal > empty_vecdouble;

                this->m_topologyHandler->applyCreateFunction(Index(size + i), t, empty_vecint, empty_vecdouble);

                if (this->p_onCreationCallback)
                {
                    this->p_onCreationCallback(Index(size + i), t, TopologyElementType(), empty_vecint, empty_vecdouble);
                }

            }
            else {
                this->m_topologyHandler->applyCreateFunction(Index(size + i), t, ancestors[i], coefs[i]);
                
                if (this->p_onCreationCallback)
                {
                    this->p_onCreationCallback(Index(size + i), t, TopologyElementType(), ancestors[i], coefs[i]);
                }
            }
        }
    }
    this->endEdit();

    // update map if needed
    addPostProcess(nbElements);
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::add(sofa::Size nbElements,
    const sofa::type::vector< TopologyElementType >&,
    const sofa::type::vector<sofa::type::vector<Index> >& ancestors,
    const sofa::type::vector<sofa::type::vector<SReal> >& coefs)
{
    this->add(nbElements, ancestors, coefs);
}


template <typename TopologyElementType, typename VecT>
void TopologySubsetData <TopologyElementType, VecT>::add(const sofa::type::vector<Index>& index,
    const sofa::type::vector< TopologyElementType >& elems,
    const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
    const sofa::type::vector< sofa::type::vector< SReal > >& coefs,
    const sofa::type::vector< AncestorElem >& ancestorElems)
{
    SOFA_UNUSED(elems);
    SOFA_UNUSED(ancestorElems);
    this->add(index.size(), ancestors, coefs);
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
    container_type& data = *(this->beginEdit());
    
    unsigned int cptDone = 0;
    Index last = data.size() - 1;

    // check for each element remove if it concern this subsetData
    for (Index idRemove : index)
    {
        Index idElem = sofa::InvalidID;
        
        idElem = this->indexOfElement(idRemove);
        if (idElem != sofa::InvalidID) // index in the map, need to update the subsetData
        {
            if (this->m_topologyHandler)
            {
                this->m_topologyHandler->applyDestroyFunction(idElem, data[idElem]);
            }

            if (this->p_onDestructionCallback)
            {
                this->p_onDestructionCallback(idElem, data[idElem]);
            }

            this->swap(idElem, last);
            cptDone++;
            if (last == 0)
                break;
            else
                --last;
        }

        // need to check if lastIndex in the map        
        idElem = this->indexOfElement(this->m_lastElementIndex);
        if (idElem != sofa::InvalidID)
        {
            updateLastIndex(idElem, idRemove);
        }
        this->m_lastElementIndex--;
    }

    if (cptDone != 0)
    {
        sofa::Size nbElements = 0;
        if (cptDone < data.size()) {
            nbElements = data.size() - cptDone;
        }
        data.resize(nbElements);
        removePostProcess(nbElements);
    }
    this->endEdit();
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
void TopologySubsetData <TopologyElementType, VecT>::addPostProcess(sofa::Size nbElements)
{
    for (unsigned int i = 0; i < nbElements; ++i)
    {
        this->m_lastElementIndex++;
        m_map2Elements.push_back(this->m_lastElementIndex);
    }
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
