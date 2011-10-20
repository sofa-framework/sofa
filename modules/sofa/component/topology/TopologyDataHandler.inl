/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_INL
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_INL

#include <sofa/component/topology/TopologyDataHandler.h>


namespace sofa
{

namespace component
{

namespace topology
{

///////////////////// Private functions on TopologyDataHandler changes /////////////////////////////
/*template <typename TopologyElementType, typename VecT>
bool TopologyDataHandler <TopologyElementType, VecT>::isTopologyDataRegistered()
{
    if (m_topologyData)
        return true;
    else
        return false;
}
*/

template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::swap( unsigned int i1, unsigned int i2 )
{
    container_type& data = *(m_topologyData->beginEdit());
    value_type tmp = data[i1];
    data[i1] = data[i2];
    data[i2] = tmp;
    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::add(unsigned int nbElements,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    // Using default values
    container_type& data = *(m_topologyData->beginEdit());
    unsigned int i0 = data.size();
    data.resize(i0+nbElements);

    for (unsigned int i = 0; i < nbElements; ++i)
    {
        value_type& t = data[i0+i];
        if (ancestors.empty() || coefs.empty())
        {
            const sofa::helper::vector< unsigned int > empty_vecint;
            const sofa::helper::vector< double > empty_vecdouble;
            this->applyCreateFunction( i0+i, t, empty_vecint, empty_vecdouble);
        }
        else
            this->applyCreateFunction( i0+i, t, ancestors[i], coefs[i] );
    }
    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::add(unsigned int nbElements,
        const sofa::helper::vector< TopologyElementType >& elems,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    // Using default values
    container_type& data = *(m_topologyData->beginEdit());
    unsigned int i0 = data.size();
    data.resize(i0+nbElements);

    for (unsigned int i = 0; i < nbElements; ++i)
    {
        value_type& t = data[i0+i];
        if (ancestors.empty() || coefs.empty())
        {
            const sofa::helper::vector< unsigned int > empty_vecint;
            const sofa::helper::vector< double > empty_vecdouble;
            this->applyCreateFunction( i0+i, t, elems[i], empty_vecint, empty_vecdouble);
        }
        else
            this->applyCreateFunction( i0+i, t, elems[i], ancestors[i], coefs[i] );
    }
    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::move( const sofa::helper::vector<unsigned int> &indexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs)
{
    container_type& data = *(m_topologyData->beginEdit());

    for (unsigned int i = 0; i <indexList.size(); i++)
    {
        this->applyDestroyFunction( indexList[i], data[indexList[i]] );
        this->applyCreateFunction( indexList[i], data[indexList[i]], ancestors[i], coefs[i] );
    }

    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::remove( const sofa::helper::vector<unsigned int> &index )
{

    container_type& data = *(m_topologyData->beginEdit());
    unsigned int last = data.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        this->applyDestroyFunction( index[i], data[index[i]] );
        this->swap( index[i], last );
        --last;
    }

    data.resize( data.size() - index.size() );
    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::renumber( const sofa::helper::vector<unsigned int> &index )
{
    container_type& data = *(m_topologyData->beginEdit());

    container_type copy = this->getValue(); // not very efficient memory-wise, but I can see no better solution...
    for (unsigned int i = 0; i < index.size(); ++i)
        data[i] = copy[ index[i] ];

    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::addOnMovedPosition(const sofa::helper::vector<unsigned int> &indexList,
        const sofa::helper::vector<TopologyElementType> &elems)
{
    container_type& data = *(m_topologyData->beginEdit());

    // Recompute data
    sofa::helper::vector< unsigned int > ancestors;
    sofa::helper::vector< double >  coefs;
    coefs.push_back (1.0);
    ancestors.resize(1);

    for (unsigned int i = 0; i <indexList.size(); i++)
    {
        ancestors[0] = indexList[i];
        this->applyCreateFunction( indexList[i], data[indexList[i]], elems[i], ancestors, coefs );
    }
    m_topologyData->endEdit();
}



template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::removeOnMovedPosition(const sofa::helper::vector<unsigned int> &indices)
{
    container_type& data = *(m_topologyData->beginEdit());

    for (unsigned int i = 0; i <indices.size(); i++)
        this->applyDestroyFunction( indices[i], data[indices[i]] );

    m_topologyData->endEdit();

    // TODO check why this call.
    this->remove( indices );
}


} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYDATA_INL
