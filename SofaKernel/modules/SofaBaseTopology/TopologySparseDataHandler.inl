/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
    sofa::component::topology::TopologySparseDataImpl<TopologyElementType, VecT>* _topologyData = dynamic_cast<sofa::component::topology::TopologySparseDataImpl<TopologyElementType, VecT>* >(m_topologyData);

    if (!_topologyData->getSparseDataStatus())
        return;

    container_type& data = *(_topologyData->beginEdit());
    sofa::helper::vector <unsigned int>& keys = _topologyData->getMap2Elements();

    /*    unsigned int pos1, pos2;
        unsigned int cpt = 0;

        for (unsigned int i=0; i<keys.size(); ++i)
        {
            if (i1 == keys[i])
            {
                pos1 = i;
                cpt++;
            }

            if (i2 == keys[i])
            {
                pos2 = i;
                cpt++;
            }

            if (cpt == 2)
                break;
        }

        if (cpt < 2)
            return;

        value_type& t = data[pos2];
        data[pos2] = data[pos1];
        data[pos1] = t;


        //apply same change to map:
        unsigned int tmp = keys[pos2];
        keys[pos2] = keys[pos1];
        keys[pos1] = tmp;
        */

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
    sofa::component::topology::TopologySparseDataImpl<TopologyElementType, VecT>* _topologyData = dynamic_cast<sofa::component::topology::TopologySparseDataImpl<TopologyElementType, VecT>* >(m_topologyData);
    if (!_topologyData->getSparseDataStatus())
        return;

    // Using default values
    sofa::helper::vector <unsigned int>& keys = _topologyData->getMap2Elements();
    container_type& data = *(_topologyData->beginEdit());
    unsigned int size = data.size();
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
    if (!_topologyData->getSparseDataStatus())
        return;

    sofa::helper::vector <unsigned int>& keys = _topologyData->getMap2Elements();
    container_type& data = *(_topologyData->beginEdit());
    unsigned int last = data.size() -1;

    for (unsigned int i = 0; i < index.size(); ++i)
    {
        this->applyDestroyFunction( index[i], data[index[i]] );
        this->swap( index[i], last );
        --last;
    }

    data.resize( data.size() - index.size() );
    keys.resize( data.size() - index.size() );
    this->lastElementIndex = last;

    _topologyData->endEdit();
    return;
}


template <typename TopologyElementType, typename VecT>
void TopologySparseDataHandler <TopologyElementType, VecT>::renumber( const sofa::helper::vector<unsigned int>& )
{
    std::cerr << "WARNING: renumber event on topology SparseData is not yet handled" << std::endl;
}


template <typename TopologyElementType, typename VecT>
void TopologySparseDataHandler <TopologyElementType, VecT>::addOnMovedPosition(const sofa::helper::vector<unsigned int> &,
        const sofa::helper::vector<TopologyElementType> &)
{
    std::cerr << "WARNING: addOnMovedPosition event on topology SparseData is not yet handled" << std::endl;
}



template <typename TopologyElementType, typename VecT>
void TopologySparseDataHandler <TopologyElementType, VecT>::removeOnMovedPosition(const sofa::helper::vector<unsigned int> &)
{
    std::cerr << "WARNING: removeOnMovedPosition event on topology SparseData is not yet handled" << std::endl;
}




} // namespace topology

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSPARSEDATAHANDLER_INL
