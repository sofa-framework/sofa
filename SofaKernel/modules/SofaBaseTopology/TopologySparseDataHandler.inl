/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSPARSEDATAHANDLER_INL
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSPARSEDATAHANDLER_INL

#include <SofaBaseTopology/TopologySparseDataHandler.h>
#include <SofaBaseTopology/TopologySparseData.h>

namespace sofa
{

namespace component
{

namespace topology
{

///////////////////// Private functions on TopologySparseDataHandler changes /////////////////////////////
template <typename TopologyElementType, typename VecT>
void TopologySparseDataHandler <TopologyElementType, VecT>::swap( unsigned int i1, unsigned int i2 )
{
    // get access to data and its map
    sofa::component::topology::TopologySparseDataImpl<TopologyElementType, VecT>* _topologyData = dynamic_cast<sofa::component::topology::TopologySparseDataImpl<TopologyElementType, VecT>* >(m_topologyData);    
    container_type& data = *(_topologyData->beginEdit());
    sofa::helper::vector <unsigned int>& keys = _topologyData->getMap2Elements();

    value_type tmp = data[i1];
    data[i1] = data[i2];
    data[i2] = tmp;

    //apply same change to map:
    unsigned int tmp2 = keys[i1];
    keys[i1] = keys[i2];
    keys[i2] = tmp2;

    _topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologySparseDataHandler <TopologyElementType, VecT>::add(unsigned int nbElements,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    // get access to data and its map
    sofa::component::topology::TopologySparseDataImpl<TopologyElementType, VecT>* _topologyData = dynamic_cast<sofa::component::topology::TopologySparseDataImpl<TopologyElementType, VecT>* >(m_topologyData);
    if (!_topologyData->getSparseDataStatus())
        return;

    container_type& data = *(_topologyData->beginEdit());
    sofa::helper::vector <unsigned int>& keys = _topologyData->getMap2Elements();

    size_type size = data.size();
    data.resize(size+nbElements);

    for (unsigned int i = 0; i < nbElements; ++i)
    {
        value_type t;
        if (ancestors.empty() || coefs.empty())
        {
            const sofa::helper::vector< unsigned int > empty_vecint;
            const sofa::helper::vector< double > empty_vecdouble;
            this->applyCreateFunction( size+i, t, empty_vecint, empty_vecdouble);
        }
        else
            this->applyCreateFunction( size+i, t, ancestors[i], coefs[i] );

        // Incremente the total number of edges in topology
        this->lastElementIndex++;
        keys.push_back(this->lastElementIndex);
    }

    _topologyData->endEdit();
    return;
}


template <typename TopologyElementType, typename VecT>
void TopologySparseDataHandler <TopologyElementType, VecT>::add(unsigned int nbElements,
        const sofa::helper::vector< TopologyElementType >& ,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    this->add(nbElements, ancestors, coefs);
}


template <typename TopologyElementType, typename VecT>
void TopologySparseDataHandler <TopologyElementType, VecT>::move( const sofa::helper::vector<unsigned int> &,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ,
        const sofa::helper::vector< sofa::helper::vector< double > >& )
{
    msg_warning("TopologySparseDataHandler") << "Move event on topology SparseData is not yet handled." ;
}


template <typename TopologyElementType, typename VecT>
void TopologySparseDataHandler <TopologyElementType, VecT>::remove( const sofa::helper::vector<unsigned int> &index )
{
    sofa::component::topology::TopologySparseDataImpl<TopologyElementType, VecT>* _topologyData = dynamic_cast<sofa::component::topology::TopologySparseDataImpl<TopologyElementType, VecT>* >(m_topologyData);

    // get the sparseData map
    sofa::helper::vector <unsigned int>& keys = _topologyData->getMap2Elements();
    container_type& data = *(_topologyData->beginEdit());
    size_type last = data.size() -1;

    // check for each element remove if it concern this sparseData
    unsigned int cptDone = 0;
    for (size_type i = 0; i < index.size(); ++i)
    {
        unsigned int elemId = index[i];
        unsigned int id = _topologyData->indexOfElement(elemId);

        if (id == sofa::core::topology::Topology::InvalidID)
            continue;

        cptDone++;
        this->applyDestroyFunction( id, data[id] );
        this->swap( id, last );
        --last;
    }

    data.resize( data.size() - cptDone );
    keys.resize( keys.size() - cptDone );
    this->lastElementIndex = last;

    _topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologySparseDataHandler <TopologyElementType, VecT>::renumber( const sofa::helper::vector<unsigned int>& )
{
    msg_warning("TopologySparseDataHandler") << "renumber event on topology SparseData is not yet handled" ;
}


template <typename TopologyElementType, typename VecT>
void TopologySparseDataHandler <TopologyElementType, VecT>::addOnMovedPosition(const sofa::helper::vector<unsigned int> &,
        const sofa::helper::vector<TopologyElementType> &)
{
    msg_warning("TopologySparseDataHandler") << "addOnMovedPosition event on topology SparseData is not yet handled" ;
}



template <typename TopologyElementType, typename VecT>
void TopologySparseDataHandler <TopologyElementType, VecT>::removeOnMovedPosition(const sofa::helper::vector<unsigned int> &)
{
    msg_warning("TopologySparseDataHandler") << "removeOnMovedPosition event on topology SparseData is not yet handled" ;
}



} // namespace topology

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSPARSEDATAHANDLER_INL
