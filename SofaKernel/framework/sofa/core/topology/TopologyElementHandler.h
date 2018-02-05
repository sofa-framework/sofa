/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_CORE_TOPOLOGY_TOPOLOGYELEMENTHANDLER_H
#define SOFA_CORE_TOPOLOGY_TOPOLOGYELEMENTHANDLER_H

#include <sofa/core/topology/TopologyHandler.h>

namespace sofa
{

namespace core
{

namespace topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////   Generic Handler Implementation   ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing Edge related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: Edges added, removed, fused, renumbered).
*/
template<class TopologyElementType>
class SOFA_CORE_API TopologyElementHandler : public sofa::core::topology::TopologyHandler
{
public:

    typedef core::topology::TopologyElementInfo<TopologyElementType> ElementInfo;
    typedef core::topology::TopologyChangeElementInfo<TopologyElementType> ChangeElementInfo;

    // Event types (EMoved* are not used for all element types, i.e. Point vs others)
    typedef typename ChangeElementInfo::EIndicesSwap    EIndicesSwap;
    typedef typename ChangeElementInfo::ERenumbering    ERenumbering;
    typedef typename ChangeElementInfo::EAdded          EAdded;
    typedef typename ChangeElementInfo::ERemoved        ERemoved;
    typedef typename ChangeElementInfo::EMoved          EMoved;
    typedef typename ChangeElementInfo::EMoved_Removing EMoved_Removing;
    typedef typename ChangeElementInfo::EMoved_Adding   EMoved_Adding;
    typedef typename ChangeElementInfo::AncestorElem    AncestorElem;

    TopologyElementHandler() : TopologyHandler() {}

    virtual ~TopologyElementHandler() {}


    using TopologyHandler::ApplyTopologyChange;

    /// Apply swap between indices elements.
    virtual void ApplyTopologyChange(const EIndicesSwap* event);
    /// Apply adding elements.
    virtual void ApplyTopologyChange(const EAdded* event);
    /// Apply removing elements.
    virtual void ApplyTopologyChange(const ERemoved* event);
    /// Apply renumbering on elements.
    virtual void ApplyTopologyChange(const ERenumbering* event);
    /// Apply moving elements.
    virtual void ApplyTopologyChange(const EMoved* event);
    /// Apply adding function on moved elements.
    virtual void ApplyTopologyChange(const EMoved_Adding* event);
    /// Apply removing function on moved elements.
    virtual void ApplyTopologyChange(const EMoved_Removing* event);

protected:
    /// Swaps values at indices i1 and i2.
    virtual void swap( unsigned int /*i1*/, unsigned int /*i2*/ ) {}

    /// Reorder the values.
    virtual void renumber( const sofa::helper::vector<unsigned int> &/*index*/ ) {}

    /// Add some values. Values are added at the end of the vector.
    /// This is the (old) version, to be deprecated in favor of the next method
    virtual void add( unsigned int /*nbElements*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &/*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}
    
    /// Add some values. Values are added at the end of the vector.
    /// This (new) version gives more information for element indices and ancestry
    virtual void add( const sofa::helper::vector<unsigned int> & index,
            const sofa::helper::vector< TopologyElementType >& elems,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
            const sofa::helper::vector< sofa::helper::vector< double > >& coefs,
            const sofa::helper::vector< AncestorElem >& /*ancestorElems*/
            )
    {
        // call old method by default
        add((unsigned int)index.size(), elems, ancestors, coefs);
    }

    /// Remove the values corresponding to the ELement removed.
    virtual void remove( const sofa::helper::vector<unsigned int> & /*index */) {}

    /// Move a list of points
    virtual void move( const sofa::helper::vector<unsigned int> & /*indexList*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}

    /// Add Element after a displacement of vertices, ie. add element based on previous position topology revision.
    virtual void addOnMovedPosition(const sofa::helper::vector<unsigned int> &/*indexList*/,
            const sofa::helper::vector< TopologyElementType > & /*elems*/) {}

    /// Remove Element after a displacement of vertices, ie. add element based on previous position topology revision.
    virtual void removeOnMovedPosition(const sofa::helper::vector<unsigned int> &/*indices*/) {}

};


} // namespace topology

} // namespace core

} // namespace sofa


#endif // SOFA_CORE_TOPOLOGY_TOPOLOGYELEMENTHANDLER_H
