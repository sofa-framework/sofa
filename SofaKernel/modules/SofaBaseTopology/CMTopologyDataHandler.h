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
#ifndef SOFA_COMPONENT_CM_TOPOLOGY_TOPOLOGYDATAHANDLER_H
#define SOFA_COMPONENT_CM_TOPOLOGY_TOPOLOGYDATAHANDLER_H
#include "config.h"

#include <sofa/core/topology/CMTopologyElementHandler.h>
#include <sofa/core/topology/CMBaseTopologyData.h>


namespace sofa
{

namespace component
{

namespace cm_topology
{


////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing Edge related data. Automatically manages topology changes.**/

template< class TopologyElementType, class T>
class TopologyDataHandler : public sofa::core::cm_topology::TopologyElementHandler< TopologyElementType >
{
public:
	using container_type = typename core::cm_topology::BaseTopologyData<T>::Attribute;
    typedef T value_type;

    /// size_type
    typedef typename std::size_t size_type;
    /// reference to a value (read-write)
    typedef typename std::add_lvalue_reference<T>::type reference;
    /// const reference to a value (read only)
    typedef typename std::add_lvalue_reference<typename std::add_const<T>::type>::type const_reference;
    /// const iterator
    typedef typename container_type::const_iterator const_iterator;

    typedef sofa::core::cm_topology::TopologyElementHandler< TopologyElementType > Inherit;
    typedef typename Inherit::AncestorElem AncestorElem;

protected:
    sofa::core::cm_topology::BaseTopologyData <T>* m_topologyData;
	value_type m_defaultValue; // default value when adding an element (by set as value_type() by default)

public:
    // constructor
    TopologyDataHandler(sofa::core::cm_topology::BaseTopologyData <T>* _topologyData,
                        value_type defaultValue=value_type())
        :sofa::core::cm_topology::TopologyElementHandler< TopologyElementType >()
        , m_topologyData(_topologyData), m_defaultValue(defaultValue) {}

    bool isTopologyDataRegistered()
    {
        if(m_topologyData) return true;
        else return false;
    }

    /** Public fonction to apply creation and destruction functions */
    /// Apply removing current elementType elements
    virtual void applyDestroyFunction(TopologyElementType, value_type& ) {}

    /// Apply adding current elementType elements
    virtual void applyCreateFunction(TopologyElementType, value_type& t,
            const sofa::helper::vector< TopologyElementType > &,
            const sofa::helper::vector< double > &) {t = m_defaultValue;}

    /// WARNING NEEED TO UNIFY THIS
    /// Apply adding current elementType elements
    virtual void applyCreateFunction(value_type&t , const TopologyElementType& e,
            const sofa::helper::vector< TopologyElementType > &ancestors,
            const sofa::helper::vector< double > &coefs)
    {
        applyCreateFunction(e, t, ancestors, coefs);
    }

    virtual void applyCreateFunction(value_type&t , const TopologyElementType& e,
            const sofa::helper::vector< TopologyElementType > &ancestors,
            const sofa::helper::vector< double > &coefs,
            const AncestorElem* /*ancestorElem*/)
    {
        applyCreateFunction(e, t, ancestors, coefs);
    }
	// update the default value used during creation
	void setDefaultValue(const value_type &v) {
		m_defaultValue=v;
	}

protected:
    /// Swaps values at indices i1 and i2.
    virtual void swap( unsigned int i1, unsigned int i2 );

    /// Add some values. Values are added at the end of the vector.
    /// This (new) version gives more information for element indices and ancestry
    virtual void add(/* const sofa::helper::vector<TopologyElementType> & index,*/
            const sofa::helper::vector< TopologyElementType >& elems,
            const sofa::helper::vector< sofa::helper::vector< TopologyElementType > > &ancestorsElems,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    /// Remove the values corresponding to the Edges removed.
    virtual void remove( const sofa::helper::vector<TopologyElementType> &index );

    /// Reorder the values.
    virtual void renumber( const sofa::helper::vector<TopologyElementType> &index );

    /// Move a list of points
    virtual void move( const sofa::helper::vector<TopologyElementType> &indexList,
            const sofa::helper::vector< sofa::helper::vector<TopologyElementType > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    /// Add Element after a displacement of vertices, ie. add element based on previous position topology revision.
    virtual void addOnMovedPosition(const sofa::helper::vector< TopologyElementType > & elems);

    /// Remove Element after a displacement of vertices, ie. add element based on previous position topology revision.
    virtual void removeOnMovedPosition(const sofa::helper::vector<TopologyElementType> &indices);


};


} // namespace cm_topology

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_CM_TOPOLOGY_TOPOLOGYDATAHANDLER_H
