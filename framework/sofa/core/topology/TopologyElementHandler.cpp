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
#include <sofa/core/topology/TopologyElementHandler.h>


namespace sofa
{

namespace core
{

namespace topology
{

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
void TopologyElementHandler<Edge>::applyEdgeRenumbering(const sofa::helper::vector<unsigned int>& indices)
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
void TopologyElementHandler<Triangle>::applyTriangleRenumbering(const sofa::helper::vector<unsigned int>& indices)
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
void TopologyElementHandler<Quad>::applyQuadRenumbering(const sofa::helper::vector<unsigned int>& indices)
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
void TopologyElementHandler<Tetrahedron>::applyTetrahedronRenumbering(const sofa::helper::vector<unsigned int>& indices)
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
void TopologyElementHandler<Hexahedron>::applyHexahedronRenumbering(const sofa::helper::vector<unsigned int>& indices)
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

template class SOFA_CORE_API TopologyElementHandler<Point>;
template class SOFA_CORE_API TopologyElementHandler<Edge>;
template class SOFA_CORE_API TopologyElementHandler<Triangle>;
template class SOFA_CORE_API TopologyElementHandler<Quad>;
template class SOFA_CORE_API TopologyElementHandler<Tetrahedron>;
template class SOFA_CORE_API TopologyElementHandler<Hexahedron>;

} // namespace topology

} // namespace core

} // namespace sofa
