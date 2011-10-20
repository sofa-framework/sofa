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
template< class TopologyElementType>
class TopologyElementHandler : public sofa::core::topology::TopologyHandler
{
public:
    TopologyElementHandler() : TopologyHandler() {}

    virtual ~TopologyElementHandler() {}

protected:
    /// Swaps values at indices i1 and i2.
    virtual void swap( unsigned int /*i1*/, unsigned int /*i2*/ ) {}

    /// Reorder the values.
    virtual void renumber( const sofa::helper::vector<unsigned int> &/*index*/ ) {}

    /// Add some values. Values are added at the end of the vector.
    virtual void add( unsigned int /*nbElements*/,
            const sofa::helper::vector< TopologyElementType >& /*elems*/,
            const sofa::helper::vector< sofa::helper::vector< unsigned int > > &/*ancestors*/,
            const sofa::helper::vector< sofa::helper::vector< double > >& /*coefs*/) {}

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



////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Point Handler function redefinition   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Apply adding Point elements.
template<>
void TopologyElementHandler<Point>::applyPointCreation(const sofa::helper::vector< unsigned int >& indices,
        const sofa::helper::vector< Point >& elems,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs)
{
    this->add( indices.size(),elems, ancestors, coefs );
}

/// Apply removing Point elements.
template<>
void TopologyElementHandler<Point>::applyPointDestruction(const sofa::helper::vector<unsigned int> & indices)
{
    this->remove( indices );
}

/// Apply swap between Point indices elements.
template<>
void TopologyElementHandler<Point>::applyPointIndicesSwap(unsigned int i1, unsigned int i2 )
{
    this->swap( i1, i2 );
}

/// /// Apply renumbering on Point elements.
template<>
void TopologyElementHandler<Point>::applyPointRenumbering(const sofa::helper::vector<unsigned int>& indices)
{
    this->renumber( indices );
}

///Apply moving function on moved Point elements.
template<>
void TopologyElementHandler<Point>::applyPointMove(const sofa::helper::vector<unsigned int>& indexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs)
{
    this->move( indexList, ancestors, coefs);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Edge Handler function redefinition   //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Apply adding Edge elements.
template<>
void TopologyElementHandler<Edge>::applyEdgeCreation(const sofa::helper::vector< unsigned int >& indices,
        const sofa::helper::vector< Edge >& elems,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs)
{
    this->add( indices.size(),elems, ancestors, coefs );
}

/// Apply removing Edge elements.
template<>
void TopologyElementHandler<Edge>::applyEdgeDestruction(const sofa::helper::vector<unsigned int> & indices)
{
    this->remove( indices );
}

/// Apply swap between Edge indices elements.
template<>
void TopologyElementHandler<Edge>::applyEdgeIndicesSwap(unsigned int i1, unsigned int i2 )
{
    this->swap( i1, i2 );
}

/// Apply renumbering on Edge elements.
template<>
void TopologyElementHandler<Edge>::applyeEdgeRenumbering(const sofa::helper::vector<unsigned int>& indices)
{
    this->renumber( indices );
}

/// Apply adding function on moved Edge elements.
template<>
void TopologyElementHandler<Edge>::applyEdgeMovedCreation(const sofa::helper::vector<unsigned int>& indexList,
        const sofa::helper::vector< Edge >& elems)
{
    this->addOnMovedPosition( indexList, elems );
}

/// Apply removing function on moved Edge elements.
template<>
void TopologyElementHandler<Edge>::applyEdgeMovedDestruction(const sofa::helper::vector<unsigned int> & indices)
{
    this->removeOnMovedPosition(indices);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////   Triangle Handler function redefinition   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Apply adding Triangle elements.
template<>
void TopologyElementHandler<Triangle>::applyTriangleCreation(const sofa::helper::vector< unsigned int >& indices,
        const sofa::helper::vector< Triangle >& elems,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs)
{
    this->add( indices.size(),elems, ancestors, coefs );
}

/// Apply removing Triangle elements.
template<>
void TopologyElementHandler<Triangle>::applyTriangleDestruction(const sofa::helper::vector<unsigned int> & indices)
{
    this->remove( indices );
}

/// Apply swap between Triangle indices elements.
template<>
void TopologyElementHandler<Triangle>::applyTriangleIndicesSwap(unsigned int i1, unsigned int i2 )
{
    this->swap( i1, i2 );
}

/// Apply renumbering on Triangle elements.
template<>
void TopologyElementHandler<Triangle>::applyeTriangleRenumbering(const sofa::helper::vector<unsigned int>& indices)
{
    this->renumber( indices );
}

/// Apply adding function on moved Triangle elements.
template<>
void TopologyElementHandler<Triangle>::applyTriangleMovedCreation(const sofa::helper::vector<unsigned int>& indexList,
        const sofa::helper::vector< Triangle >& elems)
{
    this->addOnMovedPosition( indexList, elems );
}

/// Apply removing function on moved Triangle elements.
template<>
void TopologyElementHandler<Triangle>::applyTriangleMovedDestruction(const sofa::helper::vector<unsigned int> & indices)
{
    this->removeOnMovedPosition(indices);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Quad Handler function redefinition   //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Apply adding Quad elements.
template<>
void TopologyElementHandler<Quad>::applyQuadCreation(const sofa::helper::vector< unsigned int >& indices,
        const sofa::helper::vector< Quad >& elems,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs)
{
    this->add( indices.size(),elems, ancestors, coefs );
}

/// Apply removing Quad elements.
template<>
void TopologyElementHandler<Quad>::applyQuadDestruction(const sofa::helper::vector<unsigned int> & indices)
{
    this->remove( indices );
}

/// Apply swap between Quad indices elements.
template<>
void TopologyElementHandler<Quad>:: applyQuadIndicesSwap(unsigned int i1, unsigned int i2 )
{
    this->swap( i1, i2 );
}

/// Apply renumbering on Quad elements.
template<>
void TopologyElementHandler<Quad>::applyeQuadRenumbering(const sofa::helper::vector<unsigned int>& indices)
{
    this->renumber( indices );
}

/// Apply adding function on moved Quad elements.
template<>
void TopologyElementHandler<Quad>::applyQuadMovedCreation(const sofa::helper::vector<unsigned int>& indexList,
        const sofa::helper::vector< Quad >& elems)
{
    this->addOnMovedPosition( indexList, elems );
}

/// Apply removing function on moved Quad elements.
template<>
void TopologyElementHandler<Quad>::applyQuadMovedDestruction(const sofa::helper::vector<unsigned int> & indices)
{
    this->removeOnMovedPosition(indices);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////   Tetrahedron Handler function redefinition   //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Apply adding Tetrahedron elements.
template<>
void TopologyElementHandler<Tetrahedron>::applyTetrahedronCreation(const sofa::helper::vector< unsigned int >& indices,
        const sofa::helper::vector< Tetrahedron >& elems,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs)
{
    this->add( indices.size(),elems, ancestors, coefs );
}

/// Apply removing Tetrahedron elements.
template<>
void TopologyElementHandler<Tetrahedron>:: applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> & indices)
{
    this->remove( indices );
}

/// Apply swap between Tetrahedron indices elements.
template<>
void TopologyElementHandler<Tetrahedron>::applyTetrahedronIndicesSwap(unsigned int i1, unsigned int i2 )
{
    this->swap( i1, i2 );
}

/// Apply renumbering on Tetrahedron elements.
template<>
void TopologyElementHandler<Tetrahedron>::applyeTetrahedronRenumbering(const sofa::helper::vector<unsigned int>& indices)
{
    this->renumber( indices );
}

/// Apply adding function on moved Tetrahedron elements.
template<>
void TopologyElementHandler<Tetrahedron>::applyTetrahedronMovedCreation(const sofa::helper::vector<unsigned int>& indexList,
        const sofa::helper::vector< Tetrahedron >& elems)
{
    this->addOnMovedPosition( indexList, elems );
}

/// Apply removing function on moved Tetrahedron elements.
template<>
void TopologyElementHandler<Tetrahedron>::applyTetrahedronMovedDestruction(const sofa::helper::vector<unsigned int> & indices)
{
    this->removeOnMovedPosition(indices);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////   Hexahedron Handler function redefinition   ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////


/// Apply adding Hexahedron elements.
template<>
void TopologyElementHandler<Hexahedron>::applyHexahedronCreation(const sofa::helper::vector< unsigned int >& indices,
        const sofa::helper::vector< Hexahedron >& elems,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs)
{
    this->add( indices.size(),elems, ancestors, coefs );
}

/// Apply removing Hexahedron elements.
template<>
void TopologyElementHandler<Hexahedron>::applyHexahedronDestruction(const sofa::helper::vector<unsigned int> & indices)
{
    this->remove( indices );
}

/// Apply swap between Hexahedron indices elements.
template<>
void TopologyElementHandler<Hexahedron>::applyHexahedronIndicesSwap(unsigned int i1, unsigned int i2 )
{
    this->swap( i1, i2 );
}

/// Apply renumbering on Hexahedron elements.
template<>
void TopologyElementHandler<Hexahedron>::applyeHexahedronRenumbering(const sofa::helper::vector<unsigned int>& indices)
{
    this->renumber( indices );
}

/// Apply adding function on moved Hexahedron elements.
template<>
void TopologyElementHandler<Hexahedron>::applyHexahedronMovedCreation(const sofa::helper::vector<unsigned int>& indexList,
        const sofa::helper::vector< Hexahedron >& elems)
{
    this->addOnMovedPosition( indexList, elems );
}

/// Apply removing function on moved Hexahedron elements.
template<>
void TopologyElementHandler<Hexahedron>::applyHexahedronMovedDestruction(const sofa::helper::vector<unsigned int> & indices)
{
    this->removeOnMovedPosition(indices);
}



} // namespace topology

} // namespace core

} // namespace sofa


#endif // SOFA_CORE_TOPOLOGY_TOPOLOGYELEMENTHANDLER_H
