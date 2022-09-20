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

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/TopologyChange.h>

namespace sofa::core::topology
{

typedef Topology::Point            Point;
typedef Topology::Edge             Edge;
typedef Topology::Triangle         Triangle;
typedef Topology::Quad             Quad;
typedef Topology::Tetrahedron      Tetrahedron;
typedef Topology::Hexahedron       Hexahedron;


/** A class that define topological Data general methods
* 
*/
template < class T = void* >
class BaseTopologyData : public sofa::core::objectmodel::Data <T>
{
public:

    /** \copydoc Data(const BaseData::BaseInitData&) */
    explicit BaseTopologyData(const sofa::core::objectmodel::BaseData::BaseInitData& init)
        : Data<T>(init)
    {
    }

    /// Add some values. Values are added at the end of the vector.
    virtual void add(const sofa::type::vector< Topology::PointID >& ,
        const sofa::type::vector< Topology::Point >& ,
        const sofa::type::vector< sofa::type::vector< Topology::PointID > >&,
        const sofa::type::vector< sofa::type::vector< SReal > >& ,
        const sofa::type::vector< PointAncestorElem >&) {}

    /// Temporary Hack: find a way to have a generic description of topological element:
    /// add Edge
    virtual void add(const sofa::type::vector< Topology::EdgeID >&,
        const sofa::type::vector< Topology::Edge >& ,
        const sofa::type::vector< sofa::type::vector< Topology::EdgeID > >&,
        const sofa::type::vector< sofa::type::vector< SReal > >& ,
        const sofa::type::vector< EdgeAncestorElem >&) {}

    /// add Triangle
    virtual void add(const sofa::type::vector< Topology::TriangleID >&,
        const sofa::type::vector< Topology::Triangle >& ,
        const sofa::type::vector< sofa::type::vector< Topology::TriangleID > > &,
        const sofa::type::vector< sofa::type::vector< SReal > >& ,
        const sofa::type::vector< TriangleAncestorElem >&) {}

    /// add Quad & Tetrahedron
    virtual void add(const sofa::type::vector< Topology::TetrahedronID >&,
        const sofa::type::vector< Topology::Tetrahedron >& ,
        const sofa::type::vector< sofa::type::vector< Topology::TetrahedronID > > &,
        const sofa::type::vector< sofa::type::vector< SReal > >& ,
        const sofa::type::vector< TetrahedronAncestorElem >&) {}

    virtual void add(const sofa::type::vector< Topology::QuadID >&,
        const sofa::type::vector< Topology::Quad >&,
        const sofa::type::vector< sofa::type::vector< Topology::QuadID > >&,
        const sofa::type::vector< sofa::type::vector< SReal > >&,
        const sofa::type::vector< QuadAncestorElem >&) {}

    /// add Hexahedron
    virtual void add(const sofa::type::vector< Topology::HexahedronID >&,
        const sofa::type::vector< Topology::Hexahedron >& ,
        const sofa::type::vector< sofa::type::vector< Topology::HexahedronID > > &,
        const sofa::type::vector< sofa::type::vector< SReal > >& ,
        const sofa::type::vector< HexahedronAncestorElem >&) {}


    /// Remove the values corresponding to the points removed.
    virtual void remove( const sofa::type::vector<unsigned int>& ) {}

    /// Swaps values at indices i1 and i2.
    virtual void swap( unsigned int , unsigned int ) {}

    /// Reorder the values.
    virtual void renumber( const sofa::type::vector<unsigned int>& ) {}

    /// Move a list of points
    virtual void move( const sofa::type::vector<unsigned int>& ,
            const sofa::type::vector< sofa::type::vector< unsigned int > >& ,
            const sofa::type::vector< sofa::type::vector< SReal > >& ) {}

    sofa::core::topology::BaseMeshTopology* getTopology()
    {
        return m_topology;
    }

    /// to handle PointSubsetData
    void setDataSetArraySize(const Index s) 
    { 
        m_lastElementIndex = (s == 0) ? sofa::InvalidID : s-1; 
    }

    /// Return the last element index of the topolgy buffer this Data is linked to. @sa m_lastElementIndex
    virtual Index getLastElementIndex() const { return m_lastElementIndex; }

protected:
    /// Pointer to the Topology this TopologyData is depending on
    sofa::core::topology::BaseMeshTopology* m_topology = nullptr;

    /** to handle properly the removal of items, the container must keep the last element index and update it during operations (add/remove).
    * Note that this index is mandatory and can't be retrieved directly from the topology in the case when several topology events are queued.
    * i.e: If 2 removalElements events are queued, the second event still point to a topology not yet updated by the first event. 
    */
    Index m_lastElementIndex = 0;
};


} // namespace sofa::core::topology
