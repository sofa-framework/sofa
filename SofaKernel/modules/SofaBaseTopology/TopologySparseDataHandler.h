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
#ifndef SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSPARSEDATAHANDLER_H
#define SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSPARSEDATAHANDLER_H
#include "config.h"

#include <sofa/core/topology/TopologyElementHandler.h>
#include <sofa/core/topology/BaseTopologyData.h>


namespace sofa
{

namespace component
{

namespace topology
{

// Define topology elements

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing Edge related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: Edges added, removed, fused, renumbered).
*/

template< class TopologyElementType, class VecT >
class TopologySparseDataHandler : public sofa::core::topology::TopologyElementHandler< TopologyElementType >
{
public:
    typedef VecT container_type;
    typedef typename container_type::value_type value_type;

    /// size_type
    typedef typename container_type::size_type size_type;
    /// reference to a value (read-write)
    typedef typename container_type::reference reference;
    /// const reference to a value (read only)
    typedef typename container_type::const_reference const_reference;
    /// const iterator
    typedef typename container_type::const_iterator const_iterator;
    /// iterator
    typedef typename container_type::iterator iterator;
	
    typedef sofa::core::topology::TopologyElementHandler< TopologyElementType > Inherit;
    typedef typename Inherit::AncestorElem AncestorElem;

protected:
    sofa::core::topology::BaseTopologyData <VecT>* m_topologyData;
	value_type m_defaultValue; // default value when adding an element (by set as value_type() by default)

public:
    // constructor
    TopologySparseDataHandler(sofa::core::topology::BaseTopologyData <VecT>* _topologyData,value_type defaultValue=value_type())
        : sofa::core::topology::TopologyElementHandler < TopologyElementType >()
        , m_topologyData(_topologyData), m_defaultValue(defaultValue) {}

    bool isTopologyDataRegistered() {return m_topologyData != 0;}

    /** Public fonction to apply creation and destruction functions */
    /// Apply removing current elementType elements
    virtual void applyDestroyFunction(unsigned int, value_type& ) {}
	    /// Apply adding current elementType elements
    virtual void applyCreateFunction(unsigned int, value_type& t,
            const sofa::helper::vector< unsigned int > &,
            const sofa::helper::vector< double > &) {t = m_defaultValue;}

    /// WARNING NEEED TO UNIFY THIS
    /// Apply adding current elementType elements
    virtual void applyCreateFunction(unsigned int i, value_type&t , const TopologyElementType& ,
            const sofa::helper::vector< unsigned int > &ancestors,
            const sofa::helper::vector< double > &coefs)
    {
        applyCreateFunction(i, t, ancestors, coefs);
    }

    virtual void applyCreateFunction(unsigned int i, value_type&t , const TopologyElementType& e,
            const sofa::helper::vector< unsigned int > &ancestors,
            const sofa::helper::vector< double > &coefs,
            const AncestorElem* /*ancestorElem*/)
    {
        applyCreateFunction(i, t, e, ancestors, coefs);
    }
	// update the default value used during creation
	void setDefaultValue(const value_type &v) {
		m_defaultValue=v;
	}


protected:
    /// Swaps values at indices i1 and i2.
    virtual void swap( unsigned int i1, unsigned int i2 );

    using core::topology::TopologyElementHandler< TopologyElementType >::add;
    /// Add some values. Values are added at the end of the vector.
    virtual void add( unsigned int nbElements,
            const sofa::helper::vector< TopologyElementType >& ,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    virtual void add( unsigned int nbElements,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    /// Remove the values corresponding to the Edges removed.
    virtual void remove( const sofa::helper::vector<unsigned int> &index );

    /// Reorder the values.
    virtual void renumber( const sofa::helper::vector<unsigned int> &index );

    /// Move a list of points
    virtual void move( const sofa::helper::vector<unsigned int> &indexList,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    /// Add Element after a displacement of vertices, ie. add element based on previous position topology revision.
    virtual void addOnMovedPosition(const sofa::helper::vector<unsigned int> &indexList,
            const sofa::helper::vector< TopologyElementType > & elems);

    /// Remove Element after a displacement of vertices, ie. add element based on previous position topology revision.
    virtual void removeOnMovedPosition(const sofa::helper::vector<unsigned int> &indices);




};


} // namespace topology

} // namespace component

} // namespace sofa


#endif // SOFA_COMPONENT_TOPOLOGY_TOPOLOGYSPARSEDATAHANDLER_H
