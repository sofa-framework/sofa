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
#include <SofaBaseTopology/TopologySparseData.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseTopology/TopologyDataEngine.inl>

namespace sofa::component::topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////// Private functions on TopologySparseData changes /////////////////////////////
template <typename TopologyElementType, typename VecT>
void TopologySparseData <TopologyElementType, VecT>::swap(Index i1, Index i2)
{
    // get access to data and its map
    container_type& data = *(this->beginEdit());
    sofa::helper::vector <Index>& keys = this->getMap2Elements();

    value_type tmp = data[i1];
    data[i1] = data[i2];
    data[i2] = tmp;

    //apply same change to map:
    Index tmp2 = keys[i1];
    keys[i1] = keys[i2];
    keys[i2] = tmp2;

    this->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologySparseData <TopologyElementType, VecT>::add(sofa::Size nbElements,
    const sofa::helper::vector<sofa::helper::vector<Index> >& ancestors,
    const sofa::helper::vector<sofa::helper::vector<double> >& coefs)
{
    // get access to data and its map
    if (!this->getSparseDataStatus())
        return;

    container_type& data = *(this->beginEdit());
    sofa::helper::vector <Index>& keys = this->getMap2Elements();

    Size size = data.size();
    data.resize(size + nbElements);

    for (unsigned int i = 0; i < nbElements; ++i)
    {
        //value_type t;
        //if (ancestors.empty() || coefs.empty())
        //{
        //    const sofa::helper::vector< Index > empty_vecint;
        //    const sofa::helper::vector< double > empty_vecdouble;
        //    this->applyCreateFunction(Index(size + i), t, empty_vecint, empty_vecdouble);
        //}
        //else
        //    this->applyCreateFunction(Index(size + i), t, ancestors[i], coefs[i]);

        // Incremente the total number of edges in topology
        this->lastElementIndex++;
        keys.push_back(this->lastElementIndex);
    }

    this->endEdit();
    return;
}


template <typename TopologyElementType, typename VecT>
void TopologySparseData <TopologyElementType, VecT>::add(sofa::Size nbElements,
    const sofa::helper::vector< TopologyElementType >&,
    const sofa::helper::vector<sofa::helper::vector<Index> >& ancestors,
    const sofa::helper::vector<sofa::helper::vector<double> >& coefs)
{
    this->add(nbElements, ancestors, coefs);
}


template <typename TopologyElementType, typename VecT>
void TopologySparseData <TopologyElementType, VecT>::move(const sofa::helper::vector<Index>&,
    const sofa::helper::vector< sofa::helper::vector< Index > >&,
    const sofa::helper::vector< sofa::helper::vector< double > >&)
{
    msg_warning("TopologySparseData") << "Move event on topology SparseData is not yet handled.";
}


template <typename TopologyElementType, typename VecT>
void TopologySparseData <TopologyElementType, VecT>::remove(const sofa::helper::vector<Index>& index)
{
    // get the sparseData map
    sofa::helper::vector <Index>& keys = this->getMap2Elements();
    container_type& data = *(this->beginEdit());
    Size last = data.size() - 1;

    // check for each element remove if it concern this sparseData
    unsigned int cptDone = 0;
    for (Size i = 0; i < index.size(); ++i)
    {
        unsigned int elemId = index[i];
        unsigned int id = this->indexOfElement(elemId);

        if (id == sofa::core::topology::Topology::InvalidID)
            continue;

        cptDone++;
        //this->applyDestroyFunction(id, data[id]);
        this->swap(id, last);
        --last;
    }

    data.resize(data.size() - cptDone);
    keys.resize(keys.size() - cptDone);
    this->lastElementIndex = last;

    this->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologySparseData <TopologyElementType, VecT>::renumber(const sofa::helper::vector<Index>&)
{
    msg_warning("TopologySparseData") << "renumber event on topology SparseData is not yet handled";
}


template <typename TopologyElementType, typename VecT>
void TopologySparseData <TopologyElementType, VecT>::addOnMovedPosition(const sofa::helper::vector<Index>&,
    const sofa::helper::vector<TopologyElementType>&)
{
    msg_warning("TopologySparseData") << "addOnMovedPosition event on topology SparseData is not yet handled";
}



template <typename TopologyElementType, typename VecT>
void TopologySparseData <TopologyElementType, VecT>::removeOnMovedPosition(const sofa::helper::vector<Index>&)
{
    msg_warning("TopologySparseData") << "removeOnMovedPosition event on topology SparseData is not yet handled";
}


} //namespace sofa::component::topology
