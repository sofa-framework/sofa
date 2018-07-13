/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CM_TOPOLOGY_TOPOLOGYDATAHANDLER_INL
#define SOFA_COMPONENT_CM_TOPOLOGY_TOPOLOGYDATAHANDLER_INL

#include <SofaBaseTopology/CMTopologyDataHandler.h>
//#include <sofa/core/topology/TopologyHandler.inl>

namespace sofa
{

namespace component
{

namespace cm_topology
{

///////////////////// Private functions on TopologyDataHandler changes /////////////////////////////
template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::swap( unsigned int i1, unsigned int i2 )
{
    container_type& data = *(m_topologyData->beginEdit());
	std::swap(data[i1], data[i2]);
    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::add(const sofa::helper::vector< TopologyElementType >& elems,
        const sofa::helper::vector<sofa::helper::vector<TopologyElementType> > &ancestorElems,
        const sofa::helper::vector<sofa::helper::vector<double> > &coefs)
{
    unsigned int nbElements = (unsigned)elems.size();
    if (nbElements == 0) return;
    // Using default values
    container_type& data = *(m_topologyData->beginEdit());
//    unsigned int i0 = (unsigned)data.size();
//    if (i0 != index[0])
//    {
//        this->m_topologyData->getOwner()->serr << "TopologyDataHandler SIZE MISMATCH in Data "
//            << this->m_topologyData->getName() << ": " << nbElements << " "
//            << core::topology::TopologyElementInfo<TopologyElementType>::name()
//            << " ADDED starting from index " << index[0]
//            << " while vector size is " << i0 << this->m_topologyData->getOwner()->sendl;
//        i0 = index[0];
//    }
//    data.resize(i0+nbElements);

    const sofa::helper::vector< TopologyElementType > empty_vecint;
    const sofa::helper::vector< double > empty_vecdouble;

	for (unsigned int i = 0; i < nbElements; ++i)
	{
		const TopologyElementType e = elems[i];
		this->applyCreateFunction(data[e], e,
								  (ancestorElems.empty() || coefs.empty()) ? empty_vecint : ancestorElems[i],
								  (ancestorElems.empty() || coefs.empty()) ? empty_vecdouble : coefs[i]);
	}
	m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::move( const sofa::helper::vector<TopologyElementType> &indexList,
        const sofa::helper::vector< sofa::helper::vector< TopologyElementType > >& ancestors,
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
void TopologyDataHandler <TopologyElementType, VecT>::remove( const sofa::helper::vector<TopologyElementType>& index )
{

	container_type& data = *(m_topologyData->beginEdit());
//	if (data.size()>0) {
//		unsigned int last = (unsigned)data.size() -1;

		for (unsigned int i = 0; i < index.size(); ++i)
		{
			this->applyDestroyFunction( index[i], data[index[i]] );
//			this->swap( index[i], last ); // TODO
//			--last;
		}

//		data.resize( data.size() - index.size() );
//	}
	m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::renumber( const sofa::helper::vector<TopologyElementType> &index )
{
    container_type& data = *(m_topologyData->beginEdit());

    container_type copy = m_topologyData->getValue(); // not very efficient memory-wise, but I can see no better solution...
    for (unsigned int i = 0; i < index.size(); ++i)
        data[i] = copy[ index[i] ];

    m_topologyData->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::addOnMovedPosition(const sofa::helper::vector<TopologyElementType> &elems)
{
    container_type& data = *(m_topologyData->beginEdit());

    // Recompute data
    sofa::helper::vector< TopologyElementType > ancestors;
    sofa::helper::vector< double >  coefs;
    coefs.push_back (1.0);
    ancestors.resize(1);

    for (unsigned int i = 0; i <elems.size(); i++)
    {
        ancestors[0] = elems[i];
        this->applyCreateFunction(data[elems[i]], elems[i], ancestors, coefs );
    }
    m_topologyData->endEdit();
}



template <typename TopologyElementType, typename VecT>
void TopologyDataHandler <TopologyElementType, VecT>::removeOnMovedPosition(const sofa::helper::vector<TopologyElementType> &elems)
{
    container_type& data = *(m_topologyData->beginEdit());

    for (unsigned int i = 0; i <elems.size(); i++)
        this->applyDestroyFunction( elems[i], data[elems[i]] );

    m_topologyData->endEdit();

    // TODO check why this call.
    //this->remove( indices );
}


} // namespace cm_topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CM_TOPOLOGY_TOPOLOGYDATAHANDLER_INL
