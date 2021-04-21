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
#include <SofaBaseTopology/config.h>

#include <SofaBaseTopology/TopologyDataHandler.h>
#include <SofaBaseTopology/TopologyData.h>

namespace sofa::component::topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/** \brief A class for storing element related data. Automatically manages topology changes.
*
* This class is a wrapper of class helper::vector that is made to take care transparently of all topology changes that might
* happen (non exhaustive list: elements added, removed, fused, renumbered).
*/
template< class TopologyElementType, class VecT>
class TopologySubsetData : public sofa::component::topology::TopologyData<TopologyElementType, VecT>
{

public:
    typedef VecT container_type;
    typedef typename container_type::value_type value_type;

    /// Size
    typedef typename container_type::Size Size;
    /// reference to a value (read-write)
    typedef typename container_type::reference reference;
    /// const reference to a value (read only)
    typedef typename container_type::const_reference const_reference;
    /// const iterator
    typedef typename container_type::const_iterator const_iterator;
    /// iterator
    typedef typename container_type::iterator iterator;
    typedef core::topology::TopologyElementInfo<TopologyElementType> ElementInfo;
    typedef core::topology::TopologyChangeElementInfo<TopologyElementType> ChangeElementInfo;
    typedef typename ChangeElementInfo::AncestorElem    AncestorElem;


    /// Constructor
    TopologySubsetData(const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data);

    /// Swaps values at indices i1 and i2.
    void swap(Index i1, Index i2) override;

    /// Add some values. Values are added at the end of the vector.
    virtual void add(sofa::Size nbElements,
        const sofa::helper::vector< TopologyElementType >&,
        const sofa::helper::vector< sofa::helper::vector< Index > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    virtual void add(sofa::Size nbElements,
        const sofa::helper::vector< sofa::helper::vector< Index > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs);

    void add(const sofa::helper::vector<Index>& index,
        const sofa::helper::vector< TopologyElementType >& elems,
        const sofa::helper::vector< sofa::helper::vector< Index > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs,
        const sofa::helper::vector< AncestorElem >& ancestorElems) override;

    /// Remove the values corresponding to the Edges removed.
    void remove(const sofa::helper::vector<Index>& index) override;

    /// Reorder the values.
    void renumber(const sofa::helper::vector<Index>& index) override;

    /// Move a list of points
    void move(const sofa::helper::vector<Index>& indexList,
        const sofa::helper::vector< sofa::helper::vector< Index > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs) override;

    /// Add Element after a displacement of vertices, ie. add element based on previous position topology revision.
    void addOnMovedPosition(const sofa::helper::vector<Index>& indexList,
        const sofa::helper::vector< TopologyElementType >& elems) override;

    /// Remove Element after a displacement of vertices, ie. add element based on previous position topology revision.
    void removeOnMovedPosition(const sofa::helper::vector<Index>& indices) override;

};


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Element Topology Data Implementation   ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< class VecT > using PointSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Point, VecT>;
template< class VecT > using EdgeSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Edge, VecT>;
template< class VecT > using TriangleSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Triangle, VecT>;
template< class VecT > using QuadSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Quad, VecT>;
template< class VecT > using TetrahedronSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Tetrahedron, VecT>;
template< class VecT > using HexahedronSubsetData = TopologySubsetData<core::topology::BaseMeshTopology::Hexahedron, VecT>;

} //namespace sofa::component::topology
